"""LogRA: Gradient-based influence estimation with LoRA."""

import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from typing import List, Union, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_utils import PretrainedConfig


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# LoRA Custom Backward for Per-Sample Gradients and FIM
# ============================================================================

class LoraBFunction(autograd.Function):
    """Custom backward function to compute per-sample gradients and FIM."""

    @staticmethod
    def forward(ctx, input, weight, module, grad_store):
        ctx.save_for_backward(input, weight)
        ctx.module = module
        ctx.grad_store = grad_store
        output = input.matmul(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        # Handle multi-dimensional inputs (e.g., [B, L, r])
        if grad_output.dim() > 2:
            B = grad_output.size(0)
            r = grad_output.size(-1)
            grad_output_flat = grad_output.view(B, -1, r)  # [B, N, r]
            input_flat = input.view(B, -1, r)  # [B, N, r]
            # Per-token outer products: [B, N, r, r]
            per_token_grad = torch.einsum("bni,bnj->bnij", grad_output_flat, input_flat)
            # Sum over tokens: [B, r, r]
            per_sample_grad = per_token_grad.sum(dim=1)
        else:
            per_sample_grad = torch.einsum("bi,bj->bij", grad_output, input)

        # Store per-sample gradient
        ctx.grad_store[id(ctx.module)] = per_sample_grad.detach()

        # Update FIM
        B = per_sample_grad.size(0)
        g_flat = per_sample_grad.view(B, -1)  # [B, r*r]
        batch_fim = torch.einsum("bi,bj->ij", g_flat, g_flat)
        ctx.module.fim.add_(batch_fim)

        # Standard gradients
        grad_weight = per_sample_grad.sum(dim=0)
        grad_input = torch.matmul(grad_output.to(weight.dtype), weight)
        return grad_input, grad_weight, None, None


class CustomLoraB(nn.Module):
    """Custom LoRA B layer that tracks per-sample gradients and FIM."""

    def __init__(self, rank: int, grad_store: dict):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rank, rank))
        self.module_id = id(self)
        self.grad_store = grad_store
        self.register_buffer('fim', torch.zeros(self.weight.numel(), self.weight.numel()))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LoraBFunction.apply(input, self.weight, self, self.grad_store)


class LoraLinear(nn.Module):
    """LoRA-augmented linear layer: output = linear(x) + C(B(A(x)))."""

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
    """Get parent module, target module, and target name from a key."""
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
    """Replace linear layers in model with LoRA-augmented versions."""
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


# ============================================================================
# LogRA Model
# ============================================================================

class LogRAModel(nn.Module):
    """LogRA influence estimator using per-sample gradients."""

    def __init__(self, model: nn.Module, rank: int = 8, only_lora: bool = False, mlp_only: bool = True):
        super().__init__()
        self.rank = rank
        self.only_lora = only_lora
        self.grad_store = {}

        # Replace linear layers with LoRA
        replace_linear_with_lora(model, rank, self.grad_store, only_lora=only_lora, mlp_only=mlp_only)
        self.model = model

        # Freeze all parameters except LoRA-B
        for name, param in self.model.named_parameters():
            if "logix_lora_B" not in name:
                param.requires_grad = False

        # Record LoRA-B module IDs
        self.lora_module_ids = []
        for module in self.model.modules():
            if isinstance(module, LoraLinear):
                self.lora_module_ids.append(module.logix_lora_B.module_id)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self) -> torch.Tensor:
        """Collect per-sample gradients after backward pass."""
        per_sample_grad_list = []
        for module_id in self.lora_module_ids:
            grad_tensor = self.grad_store.get(module_id, None)
            if grad_tensor is None:
                continue
            # Flatten: [B, r, r] -> [B, r*r]
            per_sample_grad_list.append(grad_tensor.view(grad_tensor.size(0), -1))
        if not per_sample_grad_list:
            return None
        concat_grad = torch.cat(per_sample_grad_list, dim=1)
        self.grad_store.clear()
        return concat_grad

    def get_fim(self, reset: bool = False) -> torch.Tensor:
        """Get Fisher Information Matrix from all LoRA-B modules."""
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


# ============================================================================
# High-Level LogRA Interface
# ============================================================================

class LogRA:
    """High-level interface for LogRA influence estimation.

    Example:
        >>> model = LogRA.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> train_embeds = model.encode(train_data, is_test=False)
        >>> test_embeds = model.encode(test_data, is_test=True)
        >>> scores = model.similarity(test_embeds, train_embeds)
    """

    def __init__(
        self,
        model_name: str,
        rank: int = 8,
        only_lora: bool = False,
        mlp_only: bool = True,
        torch_dtype=torch.bfloat16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize LogRA model.

        Args:
            model_name: HuggingFace model name or path
            rank: LoRA rank
            only_lora: If True, only use LoRA without base model
            mlp_only: If True, only apply LoRA to MLP layers
            torch_dtype: Model dtype
            device: Device to use
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load base LLM
        lm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )

        # Wrap with LogRA
        self.model = LogRAModel(lm_model, rank=rank, only_lora=only_lora, mlp_only=mlp_only)
        self.model = self.model.to(device)
        self.model.eval()

        self.fim = None  # Will be computed during train encoding

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        rank: int = 8,
        only_lora: bool = False,
        mlp_only: bool = True,
        torch_dtype=torch.bfloat16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Load LogRA model from pretrained LLM.

        Args:
            model_name: HuggingFace model name or path
            rank: LoRA rank (default: 8)
            only_lora: Only use LoRA without base model
            mlp_only: Only apply LoRA to MLP layers
            torch_dtype: Model dtype
            device: Device to use
        """
        return cls(
            model_name=model_name,
            rank=rank,
            only_lora=only_lora,
            mlp_only=mlp_only,
            torch_dtype=torch_dtype,
            device=device,
        )

    def _format_item(self, item: Dict[str, Any]) -> str:
        """Format data item into prompt-response pair.

        Supports multiple formats:
        - {'input': ..., 'output': ...}
        - {'instruction': ..., 'response': ...}
        - {'prompt': ..., 'targets': ...}
        - {'data': [prompt, response]}
        """
        prompt = ''
        response = ''

        # Step 1: Check 'data' field
        if 'data' in item:
            prompt = item['data'][0]
            response = item['data'][1]

        # Step 2: Check 'input' field (overwrites step 1)
        if 'input' in item:
            prompt = item.get('instruction', '') + ' ' + item['input']
            if 'output' in item:
                response = item['output']

        # Step 3: Check these in order (overwrites step 2)
        for op in ['inputs', 'prompt', 'instruction']:
            if op in item:
                prompt = item[op]
                break

        # Step 4: Check response fields
        for op in ['output', 'response', 'targets']:
            if op in item:
                response = item[op]
                break

        # Step 5: Format with prefix
        prompt = " ".join(prompt.split()[:100])
        prompt = 'Question: ' + prompt + '\nAnswer:'

        return prompt, response

    def _build_example(self, messages: List[Dict[str, str]]) -> tuple:
        """Build input_ids and labels from messages."""
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        q_ids = self.tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        labels = [-100] * len(q_ids) + ids[len(q_ids):]
        return ids[:800], labels[:800]

    def encode(
        self,
        data: List[Dict[str, Any]],
        batch_size: int = 8,
        is_test: bool = False,
        show_progress_bar: bool = True,
        damping: float = None,
    ) -> np.ndarray:
        """Encode data into gradient-based embeddings.

        Args:
            data: List of data items with 'input' and 'output' fields
            batch_size: Batch size for encoding
            is_test: If False (train), computes and stores FIM without preconditioning.
                     If True (test), applies FIM preconditioning to embeddings.
            show_progress_bar: Show progress bar
            damping: Damping factor for FIM preconditioning (auto if None)

        Returns:
            Embeddings as numpy array (train: raw gradients, test: FIM-preconditioned)
        """
        from torch.nn.utils.rnn import pad_sequence

        all_embeds = []

        # Prepare data
        data_iter = tqdm(range(0, len(data), batch_size), disable=not show_progress_bar)

        for i in data_iter:
            batch_data = data[i:i+batch_size]

            # Format and tokenize
            batch_input_ids = []
            batch_labels = []

            for item in batch_data:
                prompt, response = self._format_item(item)
                messages = [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': response}
                ]
                input_ids, labels = self._build_example(messages)
                batch_input_ids.append(torch.tensor(input_ids))
                batch_labels.append(torch.tensor(labels))

            # Pad sequences
            input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0).to(self.device)
            labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100).to(self.device)

            # Forward + backward to collect gradients
            loss = self.model(input_ids=input_ids, labels=labels, num_items_in_batch=1).loss
            loss.backward()

            # Get per-sample gradients
            per_sample_grads = self.model.step()
            if per_sample_grads is not None:
                all_embeds.append(per_sample_grads.cpu())

        embeds = torch.cat(all_embeds, dim=0)

        # Store or apply FIM
        if not is_test:
            # This is training data - compute and store FIM
            self.fim = self.model.get_fim(reset=True)
            return embeds.numpy()
        else:
            # This is test data - apply FIM preconditioning
            if self.fim is None:
                raise ValueError("Must encode training data first to compute FIM")
            embeds = self._precondition(embeds, damping=damping)
            return embeds.numpy()

    def _precondition(self, embeds: torch.Tensor, batch_size: int = 512, damping: float = None) -> torch.Tensor:
        """Apply FIM preconditioning to embeddings (verified implementation)."""
        n, k, _ = self.fim.size()
        result = []
        fim = self.fim.float()
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

    def similarity(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.ndarray,
        mode: str = 'cosine',
        batch_size: int = 2048,
    ) -> np.ndarray:
        """Compute similarity scores between query and corpus embeddings.

        Args:
            query_embeddings: Test embeddings (FIM-preconditioned from encode with is_test=True)
            corpus_embeddings: Train embeddings (raw gradients from encode with is_test=False)
            mode: Similarity mode ('cosine' or 'dot')
            batch_size: Batch size for computation

        Returns:
            Similarity matrix of shape (len(query), len(corpus))
        """
        all_scores = np.zeros(shape=(len(query_embeddings), len(corpus_embeddings)))

        query_tensor = torch.from_numpy(query_embeddings).cuda().float()
        if mode == 'cosine':
            query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)

        for i in tqdm(range(0, len(corpus_embeddings), batch_size)):
            batch = corpus_embeddings[i:i + batch_size]
            batch = torch.tensor(batch, device='cuda').float()
            if mode == 'cosine':
                batch = torch.nn.functional.normalize(batch, p=2, dim=1)
            score = torch.mm(query_tensor, batch.T).cpu().float().numpy()
            all_scores[:, i:i + batch_size] = score

        return all_scores