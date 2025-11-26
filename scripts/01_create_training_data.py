"""Stage 1: Create subset-dev pairs for training."""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airrep.data_sampler import SubsetDevSampler
from datasets import load_dataset
import json



def load_data(data_path: str):
    """Load training data."""
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif data_path.endswith('.json'):
        with open(data_path) as f:
            return json.load(f)
    else:
        # Try loading from HuggingFace datasets
        return list(load_dataset(data_path, split='train'))


def main():
    parser = argparse.ArgumentParser(description='Create subset-dev pairs for AirRep training')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data (jsonl, json, or HF dataset)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save subset-dev pairs')
    parser.add_argument('--dev_size', type=int, default=10000,
                        help='Size of development set')
    parser.add_argument('--train_size', type=int, default=100000,
                        help='Size of training set (after dev split)')
    parser.add_argument('--n_splits', type=int, default=100,
                        help='Number of cross-validation splits')
    parser.add_argument('--subset_size', type=int, default=1000,
                        help='Size of each training subset')
    parser.add_argument('--n_subsets_per_split', type=int, default=100,
                        help='Number of subsets to sample per split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 1: Creating Subset-Dev Pairs")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} examples")

    # Create sampler
    sampler = SubsetDevSampler(
        dev_size=args.dev_size,
        train_size=args.train_size,
        n_splits=args.n_splits,
        subset_size=args.subset_size,
        n_subsets_per_split=args.n_subsets_per_split,
        seed=args.seed
    )

    # Sample pairs
    print(f"\nSampling subset-dev pairs...")
    index_data = sampler.sample(data)

    # Save
    print(f"\nSaving to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        for item in index_data:
            f.write(json.dumps(item) + '\n')

    print("=" * 60)
    print(f"Created {len(index_data)} subset-dev pairs")
    print("=" * 60)


if __name__ == '__main__':
    main()