import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import json
from torch.utils.data import Dataset
from datasets import load_dataset
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import get_constant_schedule_with_warmup
from torch.optim import AdamW
import scipy.stats as stats
import torch.nn.functional as F
import torch
import copy
from torch.nn import BCELoss, BCEWithLogitsLoss
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from file_io import *

# Enable TF32 if possible
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

import os
import transformers

transformers.logging.set_verbosity_error()

def built_example(messages, tokenizer):
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        q_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        labels = [-100] * len(q_ids) + ids[len(q_ids):]
        input_ids = ids
    except:
        ids = tokenizer.encode(messages[0]['content'] + " " + messages[1]['content'])
        q_ids = tokenizer.encode(messages[0]['content'])
        labels = [-100] * len(q_ids) + ids[len(q_ids):]
        input_ids = ids
    return input_ids, labels


class TrainData(Dataset):
    def __init__(self, data, tokenizer, mem):
        self.data = data
        self.tokenizer = tokenizer
        self.mem = mem

    def __getitem__(self, index):
        item = self.data[index]

        if 'data' in item:
            prompt = item['data'][0]
            prompt = " ".join(prompt.split()[:100])
            response = item['data'][1]

        if 'input' in item:
            prompt = item['instruction'] + ' ' + item['input']

        for op in ['inputs', 'prompt', 'instruction']:
            if op in item:
                prompt = item[op]
                break

        for op in ['response', 'targets', 'output']:
            if op in item:
                response = item[op]
                break

        prompt = 'Question: ' + " ".join(prompt.split()[:256]) + '\nAnswer:'
        msg = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        input_ids, labels = built_example(msg, self.tokenizer)
        if self.mem is None:
            mem = [index]
        else:
            mem = self.mem[index]
        return torch.tensor(input_ids[:512]), torch.tensor(labels[:512]), torch.tensor(mem)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids, labels, embed = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        embed = torch.stack(embed)
        features = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(0),
            "embed": embed
        }
        return features


class GroupDataSet(Dataset):
    def __init__(self, data, tokenizer, all_scores, group_ppl, group_index, neg_num=1, topk=10, reference=None):
        self.data = data
        self.tokenizer = tokenizer
        self.all_scores = all_scores
        self.group_ppl = group_ppl
        self.group_index = group_index
        self.neg_num = neg_num
        self.topk = topk
        self.reference = reference or self.data
        self.batched_data = defaultdict(list)
        self.dev_list = []
        self.baseline_collect = defaultdict(list)
        for i in range(len(self.group_index)):
            item = self.group_index[i]
            item['loss'] = self.group_ppl[i].tolist()
            self.batched_data[item['dev_id']].append(item)
            if len(item['loss']) != len(item['dev']):
                print(len(item['loss']), len(item['dev']))
            assert len(item['loss']) == len(item['dev'])
            for did, ls in zip(item['dev'], item['loss']):
                self.baseline_collect[did].append(ls)
        self.baseline = {k: [np.mean(v), np.std(v) if np.std(v) != 0 else 1] for k, v in self.baseline_collect.items()}
        self.dev_list = list(self.batched_data.keys())
        self.example_seen = defaultdict(int)
        self.set_seen = defaultdict(int)

    def __len__(self):
        return len(self.batched_data)

    def getitem(self, index, data):
        item = data[index]

        if 'data' in item:
            prompt = item['data'][0]
            prompt = " ".join(prompt.split()[:100])
            response = item['data'][1]

        if 'input' in item:
            prompt = item['instruction'] + ' ' + item['input']

        for op in ['inputs', 'prompt', 'instruction']:
            if op in item:
                prompt = item[op]
                break

        for op in ['response', 'targets', 'output']:
            if op in item:
                response = item[op]
                break

        prompt = 'Question: ' + " ".join(prompt.split()[:256]) + '\nAnswer:'
        msg = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        input_ids, labels = built_example(msg, self.tokenizer)
        return torch.tensor(input_ids[-512:]), torch.tensor(labels[-512:])

    def get_batch(self, data_ids):
        input_list = []
        label_list = []
        for i in data_ids:
            input_ids, labels = self.getitem(i, self.data)
            input_list.append(input_ids)
            label_list.append(labels)
        return input_list, label_list

    def __getitem__(self, index):
        dev_id = self.dev_list[index]
        batch_data = self.batched_data[dev_id]
        dev_set = batch_data[0]['dev']
        select_set = []
        target = []
        for batch in batch_data:
            select_set.append(batch['select'])
            _loss = [(self.baseline[i][0] - ls) / self.baseline[i][1] for ls, i in zip(batch['loss'], dev_set)]
            target.append(_loss)
        # select_set = torch.tensor(select_set) # num_run, set_size
        target = torch.tensor(target)  # num_run, dev_size
        best_run_id = target.argmax(dim=0).tolist()  # dev_size
        best_run = [select_set[i] for i in best_run_id]  # dev_size, set_size

        dev_sampled = [i for i in range(len(dev_set))]
        np.random.shuffle(dev_sampled)
        used_run = []
        cleaned_dev_sampled = []
        dirty_dev_sampled = []
        for i in dev_sampled:
            if best_run_id[i] not in used_run:
                used_run.append(best_run_id[i])
                cleaned_dev_sampled.append(i)
            else:
                dirty_dev_sampled.append(i)
        dev_sampled = cleaned_dev_sampled + dirty_dev_sampled
        dev_sampled = dev_sampled[:self.reference]
        dev_set = [dev_set[i] for i in dev_sampled]
        best_run = [best_run[i] for i in dev_sampled]  # dev_size, set_size

        dev_input, dev_label = self.get_batch(dev_set)  # dev_size, seq_len

        select_set = best_run[:self.topk]  # topk, set_size
        set_input = []  # topk, set_size, seq_len
        for one_set in select_set:
            one_set_input, set_label = self.get_batch(one_set)
            set_input.append(one_set_input)

        target_batch = []  # dev_size, topk
        for i in dev_sampled:
            line = []
            for j in dev_sampled[:self.topk]:
                line.append(target[best_run_id[j], i].item())
            target_batch.append(line)

        return set_input, dev_input, target_batch

    def collate_fn(self, batch):
        set_input, dev_input, targets = zip(*batch)
        set_input = sum(set_input, [])
        dev_input = sum(dev_input, [])
        targets = sum(targets, [])  # dev_size, topk
        dev_input = pad_sequence(dev_input, batch_first=True, padding_value=0)
        targets = torch.tensor(targets)
        targets = torch.clip(targets, min=-10, max=10)
        features = {
            # "set_input_batch": set_input_batch,  # topk, set_size, seq_len
            "dev_input": dev_input,
            "targets": targets
        }
        for i, x in enumerate(set_input):
            features[f'set_input_{i}'] = pad_sequence(x, batch_first=True, padding_value=0)
        return features


class RandomGroupDataSet(GroupDataSet):
    def __getitem__(self, index):
        dev_id = self.dev_list[index]
        batch_data = self.batched_data[dev_id]
        dev_set = batch_data[0]['dev']

        dev_sampled = [i for i in range(len(dev_set))]
        np.random.shuffle(dev_sampled)
        dev_sampled.sort(key=lambda i: self.example_seen[dev_set[i]])
        dev_sampled = dev_sampled[:self.reference]
        dev_set = [dev_set[i] for i in dev_sampled]
        for i in dev_set:
            self.example_seen[i] += 1
        dev_input, dev_label = self.get_batch(dev_set)  # dev_size, seq_len

        set_sampled = [i for i in range(len(batch_data))]
        np.random.shuffle(set_sampled)
        set_sampled.sort(key=lambda i: self.set_seen[i + index * 100])
        set_sampled = set_sampled[:self.topk]
        for i in set_sampled:
            self.set_seen[i + index * 100] += 1

        target = []  # topk, dev_size
        set_input = []  # topk, set_size, seq_len
        for i in set_sampled:
            batch = batch_data[i]
            one_set_input, set_label = self.get_batch(batch['select'])
            set_input.append(one_set_input)
            _loss = [(self.baseline[i][0] - ls) / self.baseline[i][1] for ls, i in zip(batch['loss'], batch['dev'])]
            # _loss = [self.baseline[i][0] - ls for ls, i in zip(batch['loss'], batch['dev'])]
            _loss = [_loss[j] for j in dev_sampled]
            target.append(_loss)
        target = torch.tensor(target)
        target = target.t().tolist()  # dev_size, topk
        return set_input, dev_input, target


class ClassModel(torch.nn.Module):
    def __init__(self, model, num_class=None):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)
        sequence_lengths = torch.eq(input_ids, 0).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
        batch_size = input_ids.shape[0]
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits


def mean_pooling(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class MeanClassModel(ClassModel):
    def forward(self, input_ids, attention_mask=None, labels=None):
        attention_mask = attention_mask if attention_mask is not None else input_ids.ne(0)
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        hidden_states = mean_pooling(hidden_states, attention_mask)
        # hidden_states = F.normalize(hidden_states, p=2, dim=1)
        return hidden_states


class ProjectModel(torch.nn.Module):
    def __init__(self, model, num_class=None):
        super().__init__()
        self.model = model
        self.projector = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size,
                                         dtype=model.dtype)

    def forward(self, input_ids, attention_mask=None, labels=None):
        attention_mask = attention_mask if attention_mask is not None else input_ids.ne(0)
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        hidden_states = mean_pooling(hidden_states, attention_mask)
        hidden_states = self.projector(hidden_states)
        # hidden_states = F.normalize(hidden_states, p=2, dim=1)
        return hidden_states


class PoolingModel(torch.nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.k = 10
        ...
        # self.w1 = torch.nn.Parameter(torch.tensor(0.5))
        # self.w2 = torch.nn.Parameter(torch.tensor(0.5))
        # self.w3 = torch.nn.Parameter(torch.tensor(0.5))
        # self.gamma = torch.nn.Parameter(torch.tensor(0.0))
        # self.beta = torch.nn.Parameter(torch.tensor(1.0))
        # self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def self_attn(self, embeds):
        # embeds: set_size, d
        set_embed = embeds.mean(dim=0, keepdim=True)  # 1, d
        attn = torch.mm(set_embed, embeds.t())  # 1, set_size
        attn = F.softmax(attn.abs(), dim=-1)  # 1, set_size
        new_set_embed = torch.mm(attn, set_embed)  # 1, d
        return new_set_embed

    def forward(self, embeds, dev_embed):
        # query_states = self.q_proj(dev_embed)  # dev_size, d
        # key_states = self.k_proj(embeds)  # set_size, d
        # value_states = self.v_proj(embeds)  # set_size, d
        # attn_weights = torch.mm(query_states, key_states.t())  # dev_size, set_size
        # attn_weights = F.softmax(attn_weights, dim=-1)  # dev_size, set_size
        # attn_output = torch.mm(attn_weights, value_states)  # dev_size, d
        # dev_embed = self.o_proj(dev_embed)  # dev_size, d
        # score = (dev_embed * attn_output).sum(dim=-1, keepdim=True)

        # dev_embed: dev_size, d_model
        # embeds: set_size, d_model

        score = torch.mm(dev_embed, embeds.t())  # dev_size, set_size

        # flat_score = score.sum(dim=-1, keepdim=True)  # dev_size, 1
        # agg_embed = self.self_attn(embeds)  # 1, d
        # agg_score = torch.mm(dev_embed, agg_embed.t())  # dev_size, 1

        attn = score.abs()
        attn = F.softmax(attn, dim=-1)

        # topk_values, topk_indices = torch.topk(attn, self.k, dim=1)
        # topk_mask = torch.zeros_like(attn)
        # topk_mask.scatter_(1, topk_indices, 1.0)
        # attn = topk_mask * topk_mask

        attn_score = attn * score  # dev_size, set_size
        score = attn_score.sum(dim=-1, keepdim=True)  # dev_size, 1

        # score = (self.w1 * score + self.w2 * attn_score + self.w3 * agg_score) / (self.w1 + self.w2 + self.w3)
        # score = score.sum(dim=-1, keepdim=True)  # dev_size, 1

        # mask_attn = attn
        # topk_values, topk_indices = torch.topk(attn, self.k, dim=1)
        # mask_attn = torch.zeros_like(attn)
        # topk_values = 1
        # mask_attn.scatter_(1, topk_indices, topk_values)
        # mask_attn = mask_attn / mask_attn.sum(dim=-1, keepdim=True)
        # score = mask_attn * score  # dev_size, set_size
        # score = score.sum(dim=-1, keepdim=True)  # dev_size, 1
        return score


class ReluPoolingModel(torch.nn.Module):
    def __init__(self, hidden_size=384):
        super().__init__()
        self.ln = torch.nn.LayerNorm(1000, elementwise_affine=False)
        self.relu = torch.nn.ReLU()

    def forward(self, embeds, dev_embed):
        score = torch.mm(dev_embed, embeds.t())  # dev_size, set_size
        attn = self.ln(score)
        attn = self.relu(attn)
        attn_score = attn * score  # dev_size, set_size
        score = attn_score.sum(dim=-1, keepdim=True)  # dev_size, 1
        return score


class PooledProjectModel(torch.nn.Module):
    def __init__(self, model, num_class=None):
        super().__init__()
        self.model = model
        self.projector = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size,
                                         dtype=model.dtype)
        self.aggregator = PoolingModel(self.model.config.hidden_size)

    def aggregate(self, embeds, dev_embed):
        return self.aggregator(embeds, dev_embed)

    def encode(self, input_ids, attention_mask=None, labels=None):
        attention_mask = attention_mask if attention_mask is not None else input_ids.ne(0)
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        hidden_states = mean_pooling(hidden_states, attention_mask)
        hidden_states = self.projector(hidden_states)
        # hidden_states = F.normalize(hidden_states, p=2, dim=1)
        return hidden_states

    def forward(self, *args, mode='encode', **kwargs):
        if mode == 'encode':
            return self.encode(*args, **kwargs)
        else:
            return self.aggregate(*args, **kwargs)


def _gather_tensor(t, local_rank):
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)
    all_tensors[local_rank] = t
    return all_tensors


def gather_tensors(t, local_rank=None):
    if not dist.is_initialized():
        return t
    if local_rank is None:
        local_rank = dist.get_rank()
    return torch.cat(_gather_tensor(t, local_rank))


def dot_product(embed1, embed2, hessian=None):
    return torch.mm(embed1, embed2.t())


def pre_condition(embed, hessian):
    layer, vec, _ = hessian.size()
    embed = embed.view(embed.size(0), layer, vec)
    embed = embed.transpose(0, 1)
    pre_condition = torch.bmm(embed, hessian)
    pre_condition = pre_condition.transpose(0, 1).contiguous()
    pre_condition = pre_condition.view(-1, layer * vec)
    return pre_condition


def self_attn_mean(embeds):
    mean_v = embeds.mean(dim=0, keepdim=True)  # 1, d
    attn = torch.mm(embeds, mean_v.t())  # size, 1
    attn = attn / (attn.abs().sum() + 1)  # size, 1
    # attn = F.normalize(attn, p=1, dim=0)  # size, 1
    term2 = torch.mm(attn.t(), embeds)  # 1, d
    out = (mean_v + term2) / 2
    return out


def self_attn_score(embeds, dev_embed):
    embeds = self_attn_mean(embeds)  # 1, d
    return torch.mm(dev_embed, embeds.t())


def cross_attn_score(embeds, dev_embed):
    score = torch.mm(dev_embed, embeds.t())  # dev_size, set_size
    attn = score.softmax(dim=-1)  # dev_size, set_size
    score = attn * score
    score = score.sum(dim=-1, keepdim=True)
    return score


def mean_score(embeds, dev_embed):
    set_embed = embeds.mean(dim=0, keepdim=True)  # 1, d
    score = torch.mm(dev_embed, set_embed.t())  # dev_size, 1
    return score


def topk_score(embeds, dev_embed):
    score = torch.mm(dev_embed, embeds.t())  # dev_size, set_size
    top_k_values, _ = torch.topk(score, 10, dim=-1)
    top_k_sum = top_k_values.mean(dim=-1, keepdim=True)
    return top_k_sum


def eval_lds(scores, labels):
    lds = []
    for i in range(len(scores)):
        correlation, p_value = stats.spearmanr(scores[i], labels[i])
        correlation = 0 if np.isnan(correlation) else correlation
        lds.append(float(correlation))
    return sum(lds) / len(lds)


def broadcast_batch(batch, src=0):
    rank = dist.get_rank()
    if rank == src:
        meta = [(k, tuple(v.shape), str(v.dtype)) for k, v in batch.items()]
    else:
        meta = None
    meta_list = [meta]
    dist.broadcast_object_list(meta_list, src=src)
    meta = meta_list[0]
    new_batch = {}
    for (k, shape, dtype_str) in meta:
        dtype = getattr(torch, dtype_str.replace('torch.', ''))
        if rank == src:
            t = batch[k]
        else:
            t = torch.empty(shape, dtype=dtype, device='cuda')
        dist.broadcast(t, src=src)
        new_batch[k] = t
    return new_batch


def get_iter(data_loader):
    rank = dist.get_rank()
    if rank == 0:
        total_steps = len(data_loader)
    else:
        total_steps = None
    total_steps_list = [total_steps]
    dist.broadcast_object_list(total_steps_list, src=0)
    total_steps = total_steps_list[0]
    bar = tqdm(range(total_steps), total=total_steps, disable=rank != 0)
    if rank == 0:
        data_iter = iter(data_loader)
    else:
        data_iter = iter(range(total_steps))
    return bar, data_iter


def split_data(data, process_idx, num_processes):
    if isinstance(data, torch.Tensor):
        sublist_length, remainder = divmod(data.size(0), num_processes)
        return data[process_idx * sublist_length + min(process_idx, remainder):(process_idx + 1) * sublist_length + min(
            process_idx + 1, remainder)]
    else:
        return data


def eval_lds_and_log(scores_cpu, base_scores_cpu, targets_cpu, run,
                     loss_val, base_loss_eval, avg_loss_val, acc_val):
    lds_val = eval_lds(scores_cpu, targets_cpu)
    if base_scores_cpu is not None:
        base_lds_val = eval_lds(base_scores_cpu, targets_cpu)
    else:
        base_lds_val = 0
    run.log({
        'loss': loss_val,
        'base_loss': base_loss_eval,
        'loss_diff': loss_val - base_loss_eval,
        # 'avg_loss': avg_loss_val,
        # 'inv_loss': 1.0 / loss_val,
        # 'inv_avg': 1.0 / avg_loss_val,
        'acc': acc_val,
        'lds': lds_val,
        'base_lds': base_lds_val,
        'lds_diff': lds_val - base_lds_val
    })


def rank_net_loss(y_pred, y_true=None, weight_by_diff=False, weight_by_diff_powed=False, clip_median=False):
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    # if clip_median:
    #     abs_diff = torch.abs(true_diffs)
    #     median_val = abs_diff.median(dim=1, keepdim=True).values
    #     median_mask = abs_diff >= median_val
    #     the_mask = the_mask & median_mask

    pred_diffs = pred_diffs[the_mask]
    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    if weight is not None and clip_median:
        weight = torch.clamp(weight, min=0.0, max=10.0)

    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def main():
    from accelerate.utils import DistributedDataParallelKwargs
    import wandb
    accelerator = Accelerator(gradient_accumulation_steps=1, kwargs_handlers=[
        DistributedDataParallelKwargs(broadcast_buffers=False, find_unused_parameters=True)])
    # accelerator = Accelerator(gradient_accumulation_steps=1)

    # model_name = 'Alibaba-NLP/gte-base-en-v1.5'
    model_name = 'models/gte-small'
    # model_name = 'models/e5-small-v2'
    # model_name = 'models/TinyLlama-150M'
    # model_name = 'models/TinyLlama-1.1B'
    # model_name = 'Alibaba-NLP/gte-base-en-v1.5'
    # model_name = 'models/Qwen2.5-0.5B-Flan-8L'

    # save_path = f'models/gte-base-ranknet-weight'
    # save_path = f'models/llama-1b-soft-group'
    # save_path = f'models/llama-1b-group'
    # save_path = f'models/flan-qd-gte-k1-rank-7'
    # save_path = f'models/ultra-qd-gte-random-rank-3'
    # save_path = f'models/flan-new-3'
    save_path = f'models/ultra-gte-small-pool-1'
    # init_path = f'models/flan-qd-gte-self-rank-2'
    # init_path = f'models/flan-new-small'
    init_path = None
    init_epoch = 4

    batch_size = 1
    epochs = 50
    lr = 1e-4
    # lr = 5e-5

    topk = 32
    reference = 1000

    neg_num = 1

    data_name = 'ultrachat'

    # prepare model, tokenizer, and data loader  GTEClassModel  MeanClassModel PooledProjectModel
    model = PooledProjectModel(AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ), 768)
    model.model.is_causal = False
    # model.model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # params_to_optimize = [p for name, p in model.named_parameters() if "projector" in name or 'aggregator' in name]
    # optimizer = AdamW(params_to_optimize, lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr)

    if init_path is not None:
        model.load_state_dict(torch.load(f"{init_path}/model-{init_epoch}.pt", map_location=torch.device('cpu')))
        optimizer.load_state_dict(
            torch.load(f"{init_path}/optimizer-{init_epoch}.pt", map_location=torch.device('cpu')))

    rank = accelerator.process_index
    if rank == 0:
        # data = read_jsonl(f'data/{data_name}/train.jsonl')
        # group_ppl = torch.load(f'out/{data_name}_lds/all-loss.pt')
        # group_index = read_jsonl(f'data/{data_name}/all-train_index.jsonl')[:20000]
        data = read_jsonl(f'data/{data_name}/train.jsonl')
        group_ppl = torch.load(f'out/{data_name}_lds/loss.pt')
        group_index = read_jsonl(f'data/{data_name}/train_index.jsonl')[:10000]
        all_scores = None
        dataset = RandomGroupDataSet(data, tokenizer, all_scores, group_ppl, group_index, neg_num=neg_num, topk=topk,
                                     reference=reference)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, num_workers=64, shuffle=True)
    else:
        dataset = None
        data_loader = None
    accelerator.wait_for_everyone()
    # accelerator.print(len(data_loader), len(one_data_loader))
    # data_loader = MultiDataLoader([data_loader, one_data_loader])

    # base_model = copy.deepcopy(model)
    # base_model = accelerator.prepare(base_model)
    # base_model.eval()
    base_model = None

    model, optimizer = accelerator.prepare(model, optimizer)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)

    if accelerator.is_main_process:
        run = wandb.init(project='data_models')
        config = run.config
        config.lr = lr
        config.topk = topk
        config.save_path = save_path
        config.model_name = model_name
        config.init_path = init_path
        config.init_epoch = init_epoch
        executor = ThreadPoolExecutor(max_workers=1)

    if accelerator.is_main_process:
        os.system('sh kill_gpu.sh')

    losses = []
    for epoch in range(epochs):
        accelerator.print(f'Training {epoch}')
        accelerator.wait_for_everyone()
        model.eval()
        bar, data_iter = get_iter(data_loader)
        for it in bar:
            if dist.get_rank() == 0:
                batch = next(data_iter)
                batch = {k: v.cuda() for k, v in batch.items()}
            else:
                batch = {}
            accelerator.wait_for_everyone()
            batch = broadcast_batch(batch, src=0)
            with accelerator.accumulate(model):
                rank, tp = accelerator.process_index, accelerator.num_processes
                dev_input = split_data(batch['dev_input'], rank, tp)
                dev_embed = gather_tensors(model(input_ids=dev_input))  # dev_size, d
                with torch.no_grad():
                    if base_model is not None:
                        base_dev_embed = gather_tensors(base_model(input_ids=dev_input))  # dev_size, d

                # accelerator.print(batch['dev_input'].size())
                targets = batch['targets'].float()  # dev_size, top_k
                set_embed_batch = []
                scores_batch = []
                base_scores_batch = []
                for i in range(topk + 10):
                    if f"set_input_{i}" not in batch:
                        break
                    item = batch[f"set_input_{i}"]
                    set_input = split_data(item, rank, tp)
                    with torch.no_grad():
                        set_embed = gather_tensors(model(input_ids=set_input))  # set_size, d
                        if base_model is not None:
                            base_set_embed = gather_tensors(base_model(input_ids=set_input))
                            base_score = mean_score(base_set_embed, base_dev_embed).float()
                            base_scores_batch.append(base_score)
                    # mean_score, self_attn_score
                    # score = mean_score(set_embed, dev_embed)
                    score = model(set_embed, dev_embed, mode='pool')
                    score = score.float()  # dev_size, 1
                    scores_batch.append(score)
                    # set_embed = self_attn_mean(set_embed)
                scores_batch = torch.cat(scores_batch, dim=1)  # dev_size, topk
                if base_model is not None:
                    base_scores_batch = torch.cat(base_scores_batch, dim=1)
                    base_loss = rank_net_loss(base_scores_batch, targets, weight_by_diff=True, clip_median=True)
                else:
                    base_loss = torch.tensor(0.0)
                # accelerator.print(dev_embed.size(), set_embed_batch.size())
                # accelerator.print(scores_batch[:10])
                # accelerator.print(targets[:10])
                loss = rank_net_loss(scores_batch, targets, weight_by_diff=True, clip_median=True)
                # mse_loss = ((score - targets) ** 2).mean()
                accelerator.backward(loss)
                dev_embed = dev_embed.detach()
                for i in range(topk + 10):
                    scores_batch = scores_batch.detach()
                    if f"set_input_{i}" not in batch:
                        break
                    item = batch[f"set_input_{i}"]
                    # accelerator.print(item.size())
                    set_input = split_data(item, rank, tp)
                    set_embed = gather_tensors(model(input_ids=set_input))  # set_size, d
                    # set_embed = set_embed.mean(dim=0, keepdim=True)  # 1, d
                    # score = torch.mm(dev_embed, set_embed.t())  # dev_size, 1
                    # score = mean_score(set_embed, dev_embed)
                    score = model(set_embed, dev_embed, mode='pool')
                    score = score.float()  # dev_size, 1
                    scores_batch[:, i] = score.flatten()  # dev_size, topk
                    targets = batch['targets'].float()  # dev_size, top_k
                    # mse_loss = (score - targets) ** 2
                    # loss = mse_loss.mean()
                    loss = rank_net_loss(scores_batch, targets, weight_by_diff=True, clip_median=True)
                    accelerator.backward(loss)

                # dev_size, set_num = dev_embed.size(0), set_embed_batch.size(0)
                acc = (scores_batch.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean().item()
                # pos_rate = ((score > 0).sum() / (targets > 0).sum()).item()
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            losses.append(loss.item())
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            bar.set_postfix(loss=avg_loss, acc=acc)
            if accelerator.is_main_process:
                if base_model is not None:
                    base_scores_cpu = base_scores_batch.detach().cpu().numpy()
                else:
                    base_scores_cpu = None
                scores_cpu = scores_batch.detach().cpu().numpy()
                targets_cpu = targets.detach().cpu().numpy()
                future = executor.submit(
                    eval_lds_and_log,
                    scores_cpu,
                    base_scores_cpu,
                    targets_cpu,
                    run,
                    loss.item(),
                    base_loss.item(),
                    avg_loss,
                    acc
                )
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        os.makedirs(save_path, exist_ok=True)
        torch.save(unwrapped_model.state_dict(), f'{save_path}/model-{epoch}.pt')
        torch.save(optimizer.state_dict(), f'{save_path}/optimizer-{epoch}.pt')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    os.makedirs(save_path, exist_ok=True)
    torch.save(unwrapped_model.state_dict(), f'{save_path}/model.pt')
    torch.save(optimizer.state_dict(), f'{save_path}/optimizer.pt')
    if accelerator.is_main_process:
        executor.shutdown(wait=True)
        run.finish()


@torch.no_grad()
def compute_score(embed, mem, hessian=None, int16=False):
    all_scores = np.zeros(shape=(len(embed), len(mem)))
    if int16:
        all_scores = all_scores.astype('int16')
    bs = 4096
    embed = torch.tensor(embed).cuda().float()
    if hessian is not None:
        embed = pre_condition(embed, hessian)
    for i in tqdm(range(0, len(mem), bs), total=len(mem) // bs):
        batch = mem[i:i + bs]
        batch = torch.tensor(batch, device='cuda').float()
        score = dot_product(embed, batch).cpu().float().numpy()
        if int16:
            score = score / 10
            score = score.astype('int16')
        all_scores[:, i:i + bs] = score
    return all_scores


@torch.no_grad()
def infer_if():
    project = 'safe-ultra-r64'
    meta = json.load(open(f'out/{project}/meta.json'))
    mem = np.memmap(f'out/{project}/train.bin', dtype=meta['dtype'], mode='r', shape=tuple((meta['shape'])))
    print(meta['shape'])
    hessian = torch.load(f'out/{project}/i_hessian.pt')
    hessian = hessian.cuda()
    top_k = []
    bs = 4096
    all_scores = np.memmap(f'out/{project}/train_score.bin', dtype='int16', mode='w+', shape=(len(mem), len(mem)))
    for i in tqdm(range(0, len(mem), bs), total=len(mem) // bs):
        batch = mem[i:i + bs]
        score = compute_score(batch, mem, hessian)
        all_scores[i:i + bs] = score.astype('int16')
        all_scores.flush()


@torch.no_grad()
def encode(test_name='alpaca', start_epoch=0):
    # model_name = 'models/gte_Qwen2-7B-instruct'
    # model_name = 'intfloat/e5-small-v2'
    model_name = 'models/gte-small'
    # model_name = 'models/TinyLlama-1.1B'
    # model_name = 'HuggingFaceTB/SmolLM2-135M-Instruct'

    # save_path = model_name
    # save_path = f'models/flan-gte-small-pool-3'
    save_path = f'models/ultra-gte-small-pool-1'

    # prepare model, tokenizer, and data loader
    model_kwargs = dict(torch_dtype=torch.bfloat16) if 'bert' not in model_name else {}
    # GTEClassModel  MeanClassModel PooledProjectModel
    model = PooledProjectModel(AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # **model_kwargs
    ))
    model.model.is_causal = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if test_name == 'alpaca':
        data = [x for x in load_dataset('yahma/alpaca-cleaned')['train']] + [json.loads(x) for x in
                                                                             open('data/alpaca-output.jsonl')]
    elif test_name == 'safe':
        data = [x for x in load_dataset('PKU-Alignment/PKU-SafeRLHF-QA')['train']] + json.load(
            open('data/safe-test.json'))
    elif test_name == 'tulu':
        data = json.load(open('data/tulu-train-lite.json')) + json.load(open('data/tulu-test.json'))
    elif test_name == 'medqa':
        data = json.load(open('data/medqa/train.json')) + json.load(open('data/medqa/test.json'))[:500]
    elif test_name == 'flan':
        data = read_jsonl('data/flan/train_lite.jsonl') + read_jsonl('data/flan/test_lite.jsonl')

    dataset = TrainData(data, tokenizer, None)
    print(len(dataset))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4
    )

    model = model.cuda()
    # model = torch.nn.DataParallel(model)
    os.system('sh kill_gpu.sh')

    range_list = range(start_epoch, 100, 8)
    ckpt_list = [f"{save_path}/model-{epoch}.pt" for epoch in range_list]
    ckpt_id = [epoch for epoch in range_list]

    # ckpt_list = [f"{save_path}/model-soup.pt"]
    # ckpt_id = ['model']

    for ckpt, epoch in zip(ckpt_list, ckpt_id):
        # if os.path.exists(f'{save_path}/{test_name}-scores-{epoch}.npy'):
        #     continue
        print("Test", epoch)
        if not os.path.exists(ckpt):
            continue
        model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))

        model.eval()
        losses = []
        all_embed = []
        for batch in tqdm(data_loader):
            embed = batch.pop('embed')
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                out = model(**batch)
            all_embed.append(out.cpu())
        all_embed = torch.cat(all_embed, dim=0)
        # train_id = all_embed[:-6520]
        # test_embed = all_embed[-6520:]
        print(all_embed.size())
        # all_scores = compute_score(test_embed, train_id, None)
        # torch.save(all_embed, f"{save_path}/train_id.pt")
        # np.save(f'{save_path}/{test_name}-scores-{epoch}', all_scores)
        shape = (all_embed.size(0), all_embed.size(1))
        os.makedirs(f'{save_path}/embeddings', exist_ok=True)
        torch.save(all_embed, f'{save_path}/embeddings/{test_name}-embeddings-{epoch}.pt')
        print(f'{save_path}/embeddings/{test_name}-embeddings-{epoch}.pt')
        # torch.save(test_embed, f'{save_path}/embeddings/{test_name}-embeddings-test-{epoch}.pt')


def merge_index():
    ppl = []
    for index in tqdm(range(512), total=512):
        ppl.append(json.load(open(f'data/ultra_lds/ultra_lds/index_{index}.json')))
    json.dump(ppl, open('data/ultra_lds/index-all.json', 'w'))
    # torch.save(ppl, 'out/TinyLlama-50M/ultra_lds/loss.pt')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int, default=0)
    args = parser.parse_args()
    start = args.i
    # encode('flan', start)
    encode('safe', start)
    # encode('tulu', start)

    # merge_index()
    # infer_if()

    # main()
    # encode('flan', start)



