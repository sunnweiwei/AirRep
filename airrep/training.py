from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import AirRep


@dataclass
class SubsetLoss:
    subset: List[str]
    loss: float


class _SubsetDataset(Dataset):
    def __init__(self, pairs: List[SubsetLoss], encoder: AirRep) -> None:
        self.pairs = pairs
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        embedding = torch.tensor(self.encoder.encode(pair.subset)).mean(0)
        return embedding, torch.tensor(pair.loss, dtype=torch.float)


def generate_pairs(dataset: Iterable[str], model_name: str = "gpt2", subset_size: int = 5, samples: int = 100) -> List[SubsetLoss]:
    """Sample subsets and compute LM loss for each."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    pairs: List[SubsetLoss] = []
    data = list(dataset)
    for _ in range(samples):
        subset = random.sample(data, subset_size)
        inputs = tokenizer(subset, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        loss = out.loss.item()
        pairs.append(SubsetLoss(subset, loss))
    return pairs


def train_model(pairs: List[SubsetLoss], encoder_name: str = "Alibaba-NLP/gte-small-en-v1.5", epochs: int = 5, lr: float = 1e-3) -> nn.Module:
    """Train a simple regression head on top of AirRep embeddings."""
    encoder = AirRep(encoder_name)
    dataset = _SubsetDataset(pairs, encoder)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    regressor = nn.Linear(dataset[0][0].shape[0], 1)
    optim = torch.optim.Adam(regressor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for emb, target in loader:
            pred = regressor(emb)
            loss = loss_fn(pred.squeeze(), target)
            optim.zero_grad()
            loss.backward()
            optim.step()

    return nn.Sequential(encoder.model, regressor)


def save_pairs(pairs: List[SubsetLoss], path: str) -> None:
    data = [{"subset": p.subset, "loss": p.loss} for p in pairs]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_pairs(path: str) -> List[SubsetLoss]:
    with open(path) as f:
        data = json.load(f)
    return [SubsetLoss(**d) for d in data]
