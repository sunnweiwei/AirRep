import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import numpy as np
import os
from collections import OrderedDict


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class LoraBFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, module, grad_store):
        ctx.save_for_backward(input, weight)
        ctx.module = module  # store module pointer for FIM update.
        ctx.grad_store = grad_store
        output = input.matmul(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        # Handle the case where grad_output (and input) have extra dimensions.
        # For example, if grad_output is [B, L, r] and input is [B, L, r],
        # we first flatten the intermediate dimensions (L) and then compute the per-sample gradient.
        if grad_output.dim() > 2:
            B = grad_output.size(0)
            r = grad_output.size(-1)
            # Flatten all dimensions between batch and feature.
            grad_output_flat = grad_output.view(B, -1, r)  # shape: [B, N, r]
            input_flat = input.view(B, -1, r)  # shape: [B, N, r]
            # Compute per-token outer products: shape [B, N, r, r]
            per_token_grad = torch.einsum("bni,bnj->bnij", grad_output_flat, input_flat)
            # Sum over the token dimension so that each sample has one aggregated gradient: [B, r, r]
            per_sample_grad = per_token_grad.sum(dim=1)
        else:
            per_sample_grad = torch.einsum("bi,bj->bij", grad_output, input)

        # Store the per-sample gradient in the grad_store (using the module's id as the key).
        ctx.grad_store[id(ctx.module)] = per_sample_grad.detach()

        # Update the module’s FIM.
        # Flatten each sample's per-sample grad (from [r, r] to [r*r]).
        B = per_sample_grad.size(0)
        g_flat = per_sample_grad.view(B, -1)  # shape: [B, r*r]
        # Compute the batch contribution to the FIM (outer products summed over the batch).
        batch_fim = torch.einsum("bi,bj->ij", g_flat, g_flat)
        ctx.module.fim.add_(batch_fim)

        # The gradient with respect to weight is the sum over the batch.
        grad_weight = per_sample_grad.sum(dim=0)

        # Gradient with respect to input.
        grad_input = torch.matmul(grad_output.to(weight.dtype), weight)
        return grad_input, grad_weight, None, None


class CustomLoraB(nn.Module):
    def __init__(self, rank: int, grad_store: dict):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rank, rank))
        self.module_id = id(self)
        self.grad_store = grad_store
        # Register a buffer to accumulate the Fisher Information Matrix.
        # The FIM will be of shape (r*r, r*r).
        self.register_buffer('fim', torch.zeros(self.weight.numel(), self.weight.numel()))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LoraBFunction.apply(input, self.weight, self, self.grad_store)


class LoraLinear(nn.Module):
    def __init__(self, rank: int, linear: nn.Linear, grad_store: dict,
                 shared_module: nn.Module = None, only_lora: bool = False):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features
        self.rank = min(rank, in_features, out_features)
        self.only_lora = only_lora

        if not self.only_lora:
            self.logix_lora_A = nn.Linear(in_features, self.rank, bias=False)
            self.logix_lora_C = nn.Linear(self.rank, out_features, bias=False)
            nn.init.kaiming_uniform_(self.logix_lora_A.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.logix_lora_C.weight, a=math.sqrt(5))

        # Either use a shared module or create a new one.
        self.logix_lora_B = shared_module if shared_module is not None \
            else CustomLoraB(self.rank, grad_store)
        self._linear = linear

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.only_lora:
            return self.logix_lora_B(input_tensor)
        output = self._linear(input_tensor)
        output = output + self.logix_lora_C(self.logix_lora_B(self.logix_lora_A(input_tensor)))
        return output


def _get_submodules(model: nn.Module, key: str):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def replace_linear_with_lora(
    model: nn.Module,
    rank: int,
    grad_store: dict,
    only_lora: bool = False,
    mlp_only: bool = False,
):
    linear_module_names = [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    for name in linear_module_names:
        if mlp_only and "mlp" not in name.lower():
            continue
        current_module = model.get_submodule(name)
        parent, target, target_name = _get_submodules(model, name)
        lora_module = LoraLinear(rank, current_module, grad_store, only_lora=only_lora)
        lora_module.to(current_module.weight.dtype)
        setattr(parent, target_name, lora_module)


class LoraInfluenceEstimator(nn.Module):
    def __init__(self, model: nn.Module, rank: int = 8, only_lora: bool = False, mlp_only=True):
        super().__init__()
        self.rank = rank
        self.only_lora = only_lora

        # A dictionary to store per-sample gradients from all LoRA-B modules.
        self.grad_store = {}

        # Replace all nn.Linear layers in the model with our LoRA-augmented layers.
        replace_linear_with_lora(model, rank, self.grad_store, only_lora=only_lora, mlp_only=mlp_only)
        self.model = model

        # Freeze all parameters except those in LoRA-B branches.
        for name, param in self.model.named_parameters():
            if "logix_lora_B" not in name:
                param.requires_grad = False

        # Record a fixed ordering of LoRA-B module ids for consistent gradient concatenation.
        self.lora_module_ids = []
        for module in self.model.modules():
            if isinstance(module, LoraLinear):
                self.lora_module_ids.append(module.logix_lora_B.module_id)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self) -> torch.Tensor:
        per_sample_grad_list = []
        for module_id in self.lora_module_ids:
            grad_tensor = self.grad_store.get(module_id, None)
            if grad_tensor is None:
                continue
            # Flatten each module's per-sample grad from [B, r, r] to [B, r*r]
            per_sample_grad_list.append(grad_tensor.view(grad_tensor.size(0), -1))
        if not per_sample_grad_list:
            return None
        concat_grad = torch.cat(per_sample_grad_list, dim=1)
        self.grad_store.clear()
        return concat_grad

    def get_fim(self, reset: bool = False) -> dict:
        fim_mat = []
        for module_id in self.lora_module_ids:
            for module in self.model.modules():
                if isinstance(module, LoraLinear) and module.logix_lora_B.module_id == module_id:
                    b_module = module.logix_lora_B
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(b_module.fim, op=torch.distributed.ReduceOp.SUM)
                    fim_mat.append(b_module.fim.detach().cpu())
                    if reset:
                        b_module.fim.zero_()
                    break
        fim_mat = torch.stack(fim_mat, dim=0)
        return fim_mat


def aggregate_and_sort(embeds, all_index):
    import torch
    import torch.distributed as dist
    import numpy as np

    # Convert all_index to a CPU tensor if needed.
    if not torch.is_tensor(all_index):
        all_index = torch.tensor(all_index, dtype=torch.long)
    else:
        all_index = all_index.cpu()

    embeds = embeds.cpu()

    # If not running in distributed mode, sort locally and remove duplicates.
    if not dist.is_initialized():
        # Use NumPy to obtain unique indices with the first occurrence.
        indices_np = all_index.numpy()
        unique_indices, unique_pos = np.unique(indices_np, return_index=True)
        unique_pos = torch.tensor(unique_pos, dtype=torch.long)
        return embeds[unique_pos]

    world_size = dist.get_world_size()

    # Prepare lists to gather objects from all processes.
    embeds_list = [None for _ in range(world_size)]
    indices_list = [None for _ in range(world_size)]

    # Gather CPU tensors from all processes.
    dist.all_gather_object(embeds_list, embeds)
    dist.all_gather_object(indices_list, all_index)

    # Concatenate the gathered tensors.
    combined_embeds = torch.cat(embeds_list, dim=0)
    combined_indices = torch.cat(indices_list, dim=0)

    # Use NumPy to obtain the unique indices (first occurrence)
    indices_np = combined_indices.numpy()
    unique_indices, unique_pos = np.unique(indices_np, return_index=True)
    unique_pos = torch.tensor(unique_pos, dtype=torch.long)

    # The embeddings corresponding to the unique indices.
    sorted_embeds = combined_embeds[unique_pos]

    return sorted_embeds


def built_example(messages, tokenizer):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    q_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    labels = [-100] * len(q_ids) + ids[len(q_ids):]
    input_ids = ids
    return input_ids, labels


class TrainData(Dataset):
    def __init__(self, data, tokenizer, max_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.data[index]
        if 'data' in item:
            prompt = item['data'][0]
            response = item['data'][1]

        if 'input' in item:
            prompt = item['instruction']

        for op in ['inputs', 'prompt', 'instruction']:
            if op in item:
                prompt = item[op]
                break

        for op in ['response', 'targets', 'output']:
            if op in item:
                response = item[op]
                break

        prompt = " ".join(prompt.split()[:100])
        prompt = 'Question: ' + prompt + '\nAnswer:'
        msg = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        input_ids, labels = built_example(msg, self.tokenizer)
        return index, torch.tensor(input_ids[:800]), torch.tensor(labels[:800])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        index, input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        features = {
            "input_ids": input_ids,
            "labels": labels,
            "index": index
        }
        return features


def match(embed, train_mem, mode='cosine'):
    all_scores = np.zeros(shape=(len(embed), len(train_mem)))
    bs = 2048
    embed = embed.cuda().float()
    if mode == 'cosine':
        embed = torch.nn.functional.normalize(embed, p=2, dim=1)
    for i in tqdm(range(0, len(train_mem), bs), total=len(train_mem) // bs):
        batch = train_mem[i:i + bs]
        batch = torch.tensor(batch, device='cuda').float()
        if mode == 'cosine':
            batch = torch.nn.functional.normalize(batch, p=2, dim=1)
        score = torch.mm(embed, batch.T).cpu().float().numpy()
        all_scores[:, i:i + bs] = score
    return all_scores


def precondition(embeds, fim, batch_size=512, damping=None):
    n, k, _ = fim.size()
    result = []
    fim = fim.float()
    eye = torch.eye(k, device=fim.device, dtype=fim.dtype).unsqueeze(0)
    if damping is None:
        traces = fim.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        damping_module = 0.1 * traces / k
        damping_module = damping_module.view(n, 1, 1)
    else:
        damping_module = damping
    fim_damped = fim + damping_module * eye
    fim_inv = torch.inverse(fim_damped)

    for i in range(0, embeds.size(0), batch_size):
        batch = embeds[i:i+batch_size].view(-1, n, k)
        batch = batch.float()
        out = torch.einsum('ank,nkj->anj', batch, fim_inv)
        out = out.reshape(-1, n * k)
        result.append(out)
    return torch.cat(result, dim=0)


def main():
    accelerator = Accelerator()
    batch_size = 8

    model_name = "models/Qwen2.5-0.5B-Flan-Lite"
    save_path = f'out/fast-if-0.5-8-cos'

    lm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = LoraInfluenceEstimator(lm_model, rank=8, only_lora=False, mlp_only=True)
    # model = lm_model

    model.eval()

    data = read_jsonl('data/flan/train_lite.jsonl') + read_jsonl('data/flan/test_lite.jsonl')

    dataset = TrainData(data, tokenizer)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4
    )

    model, data_loader = accelerator.prepare(model, data_loader)
    tk0 = tqdm(data_loader, total=len(data_loader), disable=not accelerator.is_main_process)

    os.system('sh kill_gpu.sh')

    all_index = []
    embeds = []
    for batch in tk0:
        index = batch.pop('index')
        all_index.extend(index)
        loss = model(**batch, num_items_in_batch=1).loss
        accelerator.backward(loss)
        per_sample_grads = model.module.step()
        embeds.append(per_sample_grads.cpu())
    embeds = torch.cat(embeds, dim=0)
    embeds = aggregate_and_sort(embeds, all_index)
    fim = model.module.get_fim(reset=True)
    if accelerator.is_main_process:
        print(embeds.size())
        print(fim.size())
        train_embeds = embeds[:-6520]
        test_embeds = embeds[-6520:]
        test_embeds = precondition(test_embeds, fim, batch_size=512, damping=None)
        print(train_embeds.size(), test_embeds.size())
        score = match(test_embeds, train_embeds, mode='cosine')
        print(score.shape)
        os.makedirs(save_path, exist_ok=True)
        np.save(f'{save_path}/scores', score)


if __name__ == '__main__':
    main()