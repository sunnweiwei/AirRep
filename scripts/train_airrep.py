import argparse
from pathlib import Path

from airrep import AirRepTrainer, save_pairs


def main():
    parser = argparse.ArgumentParser(description="Train AirRep model")
    parser.add_argument("data", help="Path to a text file with one example per line")
    parser.add_argument("output", help="Directory to save model and pairs")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--subset_size", type=int, default=5)
    args = parser.parse_args()

    with open(args.data) as f:
        dataset = [line.strip() for line in f if line.strip()]

    model, pairs = AirRepTrainer.fit(
        dataset, subset_size=args.subset_size, samples=args.samples
    )
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_pairs(pairs, Path(args.output) / "pairs.json")
    model.save(Path(args.output) / "airrep.pt")


if __name__ == "__main__":
    main()
