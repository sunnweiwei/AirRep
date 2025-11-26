"""Stage 3: Train AirRep model on subset-dev pairs with losses."""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import os
from airrep.airrep_trainer import AirRepTrainer
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


def load_losses(loss_dir: str, index_data):
    """Load losses from stage 2."""
    all_losses = []

    for pair in index_data:
        pair_id = pair['id']
        loss_path = os.path.join(loss_dir, f'loss-{pair_id}.json')

        if not os.path.exists(loss_path):
            raise FileNotFoundError(f"Loss file not found: {loss_path}")

        with open(loss_path) as f:
            loss_data = json.load(f)
            all_losses.append(loss_data['losses'])

    # Convert to tensor
    losses_tensor = torch.tensor(all_losses, dtype=torch.float32)
    print(f"Loaded losses: {losses_tensor.shape}")
    return losses_tensor


def main():
    parser = argparse.ArgumentParser(description='Train AirRep model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--index_path', type=str, required=True,
                        help='Path to subset-dev pairs (from stage 1)')
    parser.add_argument('--loss_dir', type=str, required=True,
                        help='Directory with losses (from stage 2)')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save trained AirRep model')
    parser.add_argument('--base_model', type=str, default='thenlper/gte-small',
                        help='Base encoder model')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--topk', type=int, default=32,
                        help='Number of top subsets to use')
    parser.add_argument('--reference_size', type=int, default=1000,
                        help='Number of dev examples to use')
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 3: Train AirRep Model")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} examples")

    # Load index
    print(f"\nLoading subset-dev pairs from {args.index_path}...")
    index_data = load_index(args.index_path)
    print(f"Loaded {len(index_data)} pairs")

    # Load losses
    print(f"\nLoading losses from {args.loss_dir}...")
    group_losses = load_losses(args.loss_dir, index_data)

    # Initialize trainer
    trainer = AirRepTrainer(
        base_model=args.base_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        topk=args.topk,
        reference_size=args.reference_size,
        save_path=args.save_path,
    )

    # Train
    print("\nTraining AirRep model...")
    trainer.train(data, group_losses, index_data)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {args.save_path}")
    print("\nYou can now use it with:")
    print("  from airrep import AirRep")
    print(f"  model = AirRep.from_pretrained('{args.save_path}')")


if __name__ == '__main__':
    main()