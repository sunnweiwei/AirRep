"""Stage 2: Fine-tune models on subsets and evaluate on dev sets."""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import os
from airrep.sft_trainer import SFTTrainer
from datasets import load_dataset


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
        return list(load_dataset(data_path, split='train'))


def load_index(index_path: str):
    """Load subset-dev pairs."""
    index_data = []
    with open(index_path) as f:
        for line in f:
            index_data.append(json.loads(line))
    return index_data


def main():
    parser = argparse.ArgumentParser(description='SFT on subsets and evaluate')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--index_path', type=str, required=True,
                        help='Path to subset-dev pairs (from stage 1)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save losses')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B',
                        help='Base model name')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--gas', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for parallel processing')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs for parallel processing')
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 2: SFT Training and Evaluation")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} examples")

    # Load index
    print(f"\nLoading subset-dev pairs from {args.index_path}...")
    index_data = load_index(args.index_path)
    print(f"Loaded {len(index_data)} pairs")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = SFTTrainer(
        model_name=args.model_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gas,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Process pairs (parallelized across GPUs)
    # Each GPU processes every num_gpus-th pair starting from start_idx
    for i in range(args.start_idx, len(index_data), args.num_gpus):
        pair = index_data[i]
        pair_id = pair['id']

        # Check if already processed
        output_path = os.path.join(args.output_dir, f'loss-{pair_id}.json')
        if os.path.exists(output_path):
            print(f"\nSkipping {pair_id} (already processed)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing pair {pair_id} ({i+1}/{len(index_data)})")
        print(f"{'='*60}")

        # Get subset and dev data
        train_subset = [data[idx] for idx in pair['select']]
        dev_set = [data[idx] for idx in pair['dev']]

        print(f"Subset size: {len(train_subset)}, Dev size: {len(dev_set)}")

        # Train on subset
        print("\nTraining on subset...")
        model = trainer.train(train_subset)

        # Evaluate on dev
        print("\nEvaluating on dev set...")
        losses = trainer.evaluate(model, dev_set)

        # Save losses
        with open(output_path, 'w') as f:
            json.dump({
                'pair_id': pair_id,
                'dev_id': pair['dev_id'],
                'losses': losses,
                'mean_loss': sum(losses) / len(losses)
            }, f)

        print(f"\nSaved losses to {output_path}")
        print(f"Mean loss: {sum(losses) / len(losses):.4f}")

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Stage 2 Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()