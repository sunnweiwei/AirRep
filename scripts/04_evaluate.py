"""Evaluate AirRep model on test datasets."""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm


def softmax(a):
    """Softmax-weighted sum aggregation."""
    a = torch.tensor(a)
    attn = a.softmax(dim=-1)
    return (attn * a).sum().item()


def sum_agg(a):
    """Simple sum aggregation."""
    a = torch.tensor(a)
    return a.sum().item()


def format_item(x):
    prompt = ''
    response = ''
    if 'data' in x:
        prompt = x['data'][0]
        prompt = " ".join(prompt.split()[:100])
        response = x['data'][1]
    if 'input' in x:
        prompt = x.get('instruction', '') + ' ' + x['input']
    for op in ['inputs', 'prompt', 'instruction']:
        if op in x:
            prompt = x[op]
            break
    for op in ['response', 'targets', 'output']:
        if op in x:
            response = x[op]
            break

    prompt = 'Question: ' + " ".join(prompt.split()[:256]) + '\nAnswer:'
    return prompt + ' ' + response


def average_lds_spearman(
    score: np.ndarray,
    lds: List[Dict],
    use_softmax: bool = True,
) -> float:
    """
    Compute average LDS Spearman correlation.

    Args:
        score: Similarity scores (test_size, train_size)
        lds: List of dicts with 'train_subset' and 'test_loss'

    Returns:
        Average Spearman correlation
    """
    score = np.asarray(score, dtype=float)
    subset_indices = [np.asarray(s["train_subset"], dtype=int) for s in lds]
    labels = np.column_stack([np.asarray(s["test_loss"], dtype=float) for s in lds])
    labels = -labels  # Negate labels

    # Aggregate per subset with chosen method
    agg_fn = softmax if use_softmax else sum_agg
    desc = "Computing LDS (softmax)" if use_softmax else "Computing LDS (sum)"
    model_sums = np.column_stack([
        [agg_fn(score[i, idx]) for i in range(score.shape[0])]
        for idx in tqdm(subset_indices, desc=desc)
    ])

    spearman_vals = []
    for m_row, y_row in zip(model_sums, labels):
        res = stats.spearmanr(m_row, y_row)
        corr = getattr(res, "correlation", res[0])
        spearman_vals.append(0.0 if np.isnan(corr) else float(corr))

    return float(np.mean(spearman_vals))


def main():
    parser = argparse.ArgumentParser(description='Evaluate AirRep model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained AirRep model')
    parser.add_argument('--dataset', type=str, default='sunweiwei/airrep-test',
                        help='HuggingFace dataset for evaluation')
    parser.add_argument('--benchmark', type=str, default='flan',
                        help='Benchmark name (flan, alpaca, etc.)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for encoding')
    # Aggregation choice controlled by a single arg: 1 (softmax) or 0 (sum)
    parser.add_argument('--softmax', type=int, choices=[0, 1], default=1,
                        help='1 to use softmax-weighted aggregation, 0 to use sum')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Evaluating AirRep on {args.benchmark}")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    from airrep import AirRep
    model = AirRep.from_pretrained(args.model_path)
    print("Model loaded successfully")

    # Load datasets
    print(f"\nLoading {args.benchmark} dataset...")
    train_ds = load_dataset(args.dataset, data_files=f'{args.benchmark}/train.jsonl', split='train')
    test_ds = load_dataset(args.dataset, data_files=f'{args.benchmark}/test.jsonl', split='train')
    lds_ds = load_dataset(args.dataset, data_files=f'{args.benchmark}/lds.jsonl', split='train')

    print(f"Train examples: {len(train_ds)}")
    print(f"Test examples: {len(test_ds)}")
    print(f"LDS entries: {len(lds_ds)}")

    # Format texts
    print("\nFormatting texts...")
    train_texts = [format_item(x) for x in train_ds]
    test_texts = [format_item(x) for x in test_ds]

    # Encode
    print("\nEncoding train texts...")
    train_emb = model.encode(train_texts, batch_size=args.batch_size, show_progress_bar=True)

    print("\nEncoding test texts...")
    test_emb = model.encode(test_texts, batch_size=args.batch_size, show_progress_bar=True)

    print(f"\nTrain embeddings: {train_emb.shape}")
    print(f"Test embeddings: {test_emb.shape}")

    # Compute similarity in chunks
    print("\nComputing similarity scores...")
    score = []
    for i in tqdm(range(0, len(train_emb), 1000), desc="Computing scores"):
        score.append(test_emb @ train_emb[i:i+1000].T)
    score = np.hstack(score)

    # Evaluate LDS
    agg_name = 'softmax' if args.softmax == 1 else 'sum'
    print(f"\nEvaluating LDS Spearman correlation (agg={agg_name})...")
    spearman = average_lds_spearman(score, list(lds_ds), use_softmax=bool(args.softmax))

    print("\n" + "=" * 60)
    print(f"Results for {args.benchmark}")
    print("=" * 60)
    print(f"LDS Spearman Correlation: {spearman:.4f}")
    print("=" * 60)



if __name__ == '__main__':
    main()
