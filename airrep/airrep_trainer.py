"""AirRep trainer for training influence prediction model."""

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers.optimization import get_constant_schedule_with_warmup
from collections import defaultdict
from typing import List, Dict
import numpy as np
from scipy import stats
from itertools import product
from torch.nn import BCEWithLogitsLoss

from .modeling_airrep import AirRepModel, AirRepConfig


def format_item(item: Dict) -> tuple:
    """Format dataset item into prompt and response."""
    prompt = ''
    response = ''

    if 'data' in item:
        prompt = item['data'][0]
        prompt = " ".join(prompt.split()[:100])
        response = item['data'][1]

    if 'input' in item:
        prompt = item.get('instruction', '') + ' ' + item['input']

    for op in ['inputs', 'prompt', 'instruction']:
        if op in item:
            prompt = item[op]
            break

    for op in ['response', 'targets', 'output']:
        if op in item:
            response = item[op]
            break

    prompt = 'Question: ' + " ".join(prompt.split()[:256]) + '\nAnswer:'
    return prompt, response


def build_example(messages: List[Dict], tokenizer):
    """Build input_ids from messages."""
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    except:
        ids = tokenizer.encode(messages[0]['content'] + " " + messages[1]['content'])
    return ids


class AirRepDataset(Dataset):
    """Dataset for AirRep training."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        group_losses: torch.Tensor,
        group_index: List[Dict],
        topk: int = 32,
        reference_size: int = 1000,
        max_len: int = 512
    ):
        """
        Initialize dataset.

        Args:
            data: Full training corpus
            tokenizer: Tokenizer
            group_losses: Tensor of shape (n_pairs, dev_size) with losses
            group_index: List of subset-dev pair metadata
            topk: Number of top subsets to use
            reference_size: Number of dev examples to use
            max_len: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.group_losses = group_losses
        self.group_index = group_index
        self.topk = topk
        self.reference_size = reference_size
        self.max_len = max_len

        # Organize data by dev_id
        self.batched_data = defaultdict(list)
        for i in range(len(group_index)):
            item = group_index[i]
            item['loss'] = group_losses[i].tolist()
            self.batched_data[item['dev_id']].append(item)

        # Compute baseline statistics
        self.baseline_collect = defaultdict(list)
        for item in group_index:
            for did, ls in zip(item['dev'], item['loss']):
                self.baseline_collect[did].append(ls)

        self.baseline = {
            k: [np.mean(v), np.std(v) if np.std(v) != 0 else 1]
            for k, v in self.baseline_collect.items()
        }

        self.dev_list = list(self.batched_data.keys())
        self.example_seen = defaultdict(int)
        self.set_seen = defaultdict(int)

    def __len__(self):
        return len(self.batched_data)

    def get_item_encoding(self, index: int) -> torch.Tensor:
        """Get encoding for a single data example."""
        item = self.data[index]
        prompt, response = format_item(item)

        msg = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response}
        ]
        input_ids = build_example(msg, self.tokenizer)
        return torch.tensor(input_ids[-self.max_len:])

    def get_batch(self, data_ids: List[int]):
        """Get batch of encodings."""
        return [self.get_item_encoding(i) for i in data_ids]

    def __getitem__(self, index):
        """Get training example."""
        dev_id = self.dev_list[index]
        batch_data = self.batched_data[dev_id]
        dev_set = batch_data[0]['dev']

        # Sample dev examples (prioritize less seen)
        dev_sampled = list(range(len(dev_set)))
        np.random.shuffle(dev_sampled)
        dev_sampled.sort(key=lambda i: self.example_seen[dev_set[i]])
        dev_sampled = dev_sampled[:self.reference_size]
        dev_set = [dev_set[i] for i in dev_sampled]

        for i in dev_set:
            self.example_seen[i] += 1

        dev_input = self.get_batch(dev_set)

        # Sample subsets (prioritize less seen)
        set_sampled = list(range(len(batch_data)))
        np.random.shuffle(set_sampled)
        set_sampled.sort(key=lambda i: self.set_seen[i + index * 100])
        set_sampled = set_sampled[:self.topk]

        for i in set_sampled:
            self.set_seen[i + index * 100] += 1

        # Get subset embeddings and targets
        target = []
        set_input = []
        for i in set_sampled:
            batch = batch_data[i]
            one_set_input = self.get_batch(batch['select'])
            set_input.append(one_set_input)

            # Normalize losses
            _loss = [
                (self.baseline[did][0] - ls) / self.baseline[did][1]
                for ls, did in zip(batch['loss'], batch['dev'])
            ]
            _loss = [_loss[j] for j in dev_sampled]
            target.append(_loss)

        target = torch.tensor(target).t().tolist()  # (reference_size, topk)
        return set_input, dev_input, target

    def collate_fn(self, batch):
        """Collate batch."""
        set_input, dev_input, targets = zip(*batch)
        set_input = sum(set_input, [])  # Flatten
        dev_input = sum(dev_input, [])  # Flatten
        targets = sum(targets, [])

        dev_input = pad_sequence(dev_input, batch_first=True, padding_value=0)
        targets = torch.tensor(targets)
        targets = torch.clip(targets, min=-10, max=10)

        features = {
            "dev_input": dev_input,
            "targets": targets
        }

        # Pad each subset separately
        for i, x in enumerate(set_input):
            features[f'set_input_{i}'] = pad_sequence(x, batch_first=True, padding_value=0)

        return features


def ranknet_loss(y_pred, y_true, weight_by_diff=True, clip_median=True):
    """RankNet pairwise ranking loss."""
    document_pairs = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs]
    selected_pred = y_pred[:, document_pairs]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    mask = (true_diffs > 0) & (~torch.isinf(true_diffs))
    pred_diffs = pred_diffs[mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[mask]

    if weight is not None and clip_median:
        weight = torch.clamp(weight, min=0.0, max=10.0)

    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)


def eval_lds(scores, labels):
    """Evaluate LDS Spearman correlation."""
    lds = []
    for i in range(len(scores)):
        correlation, _ = stats.spearmanr(scores[i], labels[i])
        correlation = 0 if np.isnan(correlation) else correlation
        lds.append(float(correlation))
    return sum(lds) / len(lds)


class AirRepTrainer:
    """Trainer for AirRep model."""

    def __init__(
        self,
        base_model: str = "thenlper/gte-small",
        batch_size: int = 1,
        epochs: int = 50,
        lr: float = 1e-4,
        topk: int = 32,
        reference_size: int = 1000,
        save_path: str = "./airrep_model",
    ):
        """
        Initialize trainer.

        Args:
            base_model: Base encoder model
            batch_size: Training batch size
            epochs: Number of training epochs
            lr: Learning rate
            topk: Number of top subsets to use
            reference_size: Number of dev examples to use
            save_path: Path to save model checkpoints
        """
        self.base_model = base_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.topk = topk
        self.reference_size = reference_size
        self.save_path = save_path

        self.accelerator = Accelerator(gradient_accumulation_steps=1)

    def train(
        self,
        data: List[Dict],
        group_losses: torch.Tensor,
        group_index: List[Dict],
    ):
        """
        Train AirRep model.

        Args:
            data: Full training corpus
            group_losses: Tensor of shape (n_pairs, dev_size) with losses
            group_index: List of subset-dev pair metadata
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Create model
        base_encoder = AutoModel.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Create AirRep config from base model
        config = AirRepConfig(
            **base_encoder.config.to_dict()
        )
        config.model_type = "airrep"

        # Initialize AirRep model
        model = AirRepModel(config)
        model.bert = base_encoder  # Use pretrained encoder

        # Create dataset
        rank = self.accelerator.process_index
        if rank == 0:
            dataset = AirRepDataset(
                data,
                tokenizer,
                group_losses,
                group_index,
                topk=self.topk,
                reference_size=self.reference_size
            )
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=dataset.collate_fn,
                num_workers=64
            )
        else:
            data_loader = None

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=self.lr)
        model, optimizer = self.accelerator.prepare(model, optimizer)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)

        # Training loop
        for epoch in range(self.epochs):
            self.accelerator.print(f'Training epoch {epoch + 1}/{self.epochs}')
            model.eval()  # Keep in eval mode (no dropout)

            if rank == 0:
                losses = []
                for batch in tqdm(data_loader):
                    batch = {k: v.cuda() for k, v in batch.items()}

                    # Encode dev set
                    dev_embed = model(batch['dev_input'])

                    # Encode subsets and compute scores
                    scores_batch = []
                    for i in range(self.topk + 10):
                        if f"set_input_{i}" not in batch:
                            break

                        with torch.no_grad():
                            set_embed = model(batch[f"set_input_{i}"])

                        # Compute similarity with softmax
                        score = torch.mm(dev_embed, set_embed.t())  # (dev_size, subset_size)
                        attn = F.softmax(score.abs(), dim=-1)
                        score = (attn * score).sum(dim=-1, keepdim=True)  # (dev_size, 1)
                        scores_batch.append(score)

                    scores_batch = torch.cat(scores_batch, dim=1)  # (dev_size, topk)
                    targets = batch['targets'].float()

                    # Compute loss
                    loss = ranknet_loss(scores_batch, targets, weight_by_diff=True, clip_median=True)
                    self.accelerator.backward(loss)

                    # Re-encode with gradients
                    dev_embed = model(batch['dev_input'])
                    for i in range(self.topk + 10):
                        if f"set_input_{i}" not in batch:
                            break

                        scores_batch = scores_batch.detach()
                        set_embed = model(batch[f"set_input_{i}"])

                        score = torch.mm(dev_embed, set_embed.t())
                        attn = F.softmax(score.abs(), dim=-1)
                        score = (attn * score).sum(dim=-1, keepdim=True)
                        scores_batch[:, i] = score.flatten()

                        loss = ranknet_loss(scores_batch, targets, weight_by_diff=True, clip_median=True)
                        self.accelerator.backward(loss)

                    self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    losses.append(loss.item())

                avg_loss = sum(losses) / len(losses)
                self.accelerator.print(f'Epoch {epoch + 1} avg loss: {avg_loss:.4f}')

            # Save checkpoint
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(f'{self.save_path}/epoch-{epoch}')
                tokenizer.save_pretrained(f'{self.save_path}/epoch-{epoch}')

        # Save final model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(self.save_path)
            tokenizer.save_pretrained(self.save_path)