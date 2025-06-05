import argparse
import json
from pathlib import Path
import torch

from airrep import generate_pairs, train_model, save_pairs


def main():
    parser = argparse.ArgumentParser(description="Train AirRep model")
    parser.add_argument("data", help="Path to a text file with one example per line")
    parser.add_argument("output", help="Directory to save model and pairs")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--subset_size", type=int, default=5)
    args = parser.parse_args()

    with open(args.data) as f:
        dataset = [line.strip() for line in f if line.strip()]

    pairs = generate_pairs(dataset, subset_size=args.subset_size, samples=args.samples)
    save_pairs(pairs, Path(args.output)/"pairs.json")
    model = train_model(pairs)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(args.output)/"airrep.pt")


if __name__ == "__main__":
    main()
