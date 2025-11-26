"""SFT trainer for fine-tuning models on subsets."""

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from typing import List, Dict, Any


def format_item(item: Dict) -> tuple:
    """Format dataset item into prompt and response."""
    prompt = ''
    response = ''

    # Step 1: Check 'data' field
    if 'data' in item:
        prompt = item['data'][0]
        prompt = " ".join(prompt.split()[:100])
        response = item['data'][1]

    # Step 2: Check 'input' field (overwrites step 1)
    if 'input' in item:
        prompt = item.get('instruction', '') + ' ' + item['input']

    # Step 3: Check these in order (overwrites step 2)
    for op in ['inputs', 'prompt', 'instruction']:
        if op in item:
            prompt = item[op]
            break

    # Step 4: Check response fields
    for op in ['response', 'targets', 'output']:
        if op in item:
            response = item[op]
            break

    # Step 5: Format with prefix
    prompt = 'Question: ' + " ".join(prompt.split()[:256]) + '\nAnswer:'
    return prompt, response


def build_example(messages: List[Dict], tokenizer):
    """Build input_ids and labels from messages."""
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        q_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        labels = [-100] * len(q_ids) + ids[len(q_ids):]
        input_ids = ids
    except:
        # Fallback for tokenizers without chat template
        ids = tokenizer.encode(messages[0]['content'] + " " + messages[1]['content'])
        q_ids = tokenizer.encode(messages[0]['content'])
        labels = [-100] * len(q_ids) + ids[len(q_ids):]
        input_ids = ids
    return input_ids, labels


class SFTDataset(Dataset):
    """Dataset for SFT training."""

    def __init__(self, data: List[Dict], tokenizer, max_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.data[index]
        prompt, response = format_item(item)

        msg = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response}
        ]
        input_ids, labels = build_example(msg, self.tokenizer)

        return (
            torch.tensor(input_ids[:self.max_len]),
            torch.tensor(labels[:self.max_len])
        )

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class SFTTrainer:
    """Trainer for supervised fine-tuning on subsets."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        epochs: int = 2,
        lr: float = 2e-5,
        max_len: int = 512,
        use_flash_attn: bool = True,
    ):
        """
        Initialize SFT trainer.

        Args:
            model_name: Base model name or path
            batch_size: Training batch size
            gradient_accumulation_steps: Number of gradient accumulation steps
            epochs: Number of training epochs
            lr: Learning rate
            max_len: Maximum sequence length
            use_flash_attn: Use flash attention 2
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.use_flash_attn = use_flash_attn

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def train(self, train_data: List[Dict]) -> AutoModelForCausalLM:
        """
        Train model on training data.

        Args:
            train_data: Training examples

        Returns:
            Trained model
        """
        # Load model
        model_kwargs = {
            'torch_dtype': torch.bfloat16,
        }
        if self.use_flash_attn:
            model_kwargs['attn_implementation'] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # Create dataset
        dataset = SFTDataset(train_data, self.tokenizer, self.max_len)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=4
        )

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=self.lr)

        # Prepare for training
        model, optimizer, data_loader = self.accelerator.prepare(
            model, optimizer, data_loader
        )

        # Training loop
        for epoch in range(self.epochs):
            self.accelerator.print(f'Epoch {epoch + 1}/{self.epochs}')
            model.train()

            losses = []
            for batch in tqdm(data_loader, disable=not self.accelerator.is_main_process):
                with self.accelerator.accumulate(model):
                    out = model(**batch)
                    loss = out.loss

                    if torch.isnan(loss):
                        loss = torch.tensor(0.0, requires_grad=True).to(loss.device)
                        self.accelerator.print('NaN loss detected!')

                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            self.accelerator.print(f'Epoch {epoch + 1} avg loss: {avg_loss:.4f}')

        return self.accelerator.unwrap_model(model)

    @torch.no_grad()
    def evaluate(self, model: AutoModelForCausalLM, eval_data: List[Dict]) -> List[float]:
        """
        Evaluate model on dev data.

        Args:
            model: Model to evaluate
            eval_data: Evaluation examples

        Returns:
            Per-example losses
        """
        model.eval()

        # Create dataset
        dataset = SFTDataset(eval_data, self.tokenizer, self.max_len)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=4
        )

        losses = []
        for batch in tqdm(data_loader, disable=not self.accelerator.is_main_process):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Forward pass
            lm_logits = model(batch['input_ids']).logits
            bs = lm_logits.size(0)

            # Compute per-example loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )

            loss = loss.view(bs, -1)
            seq_len = batch['labels'].ne(-100).sum(dim=-1)
            loss = loss.sum(dim=-1) / seq_len
            loss[seq_len == 0] = 10.0

            losses.extend(loss.cpu().tolist())

        return losses