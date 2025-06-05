import argparse
from airrep import AirRep, evaluate_lds, load_pairs
import torch


def main():
    parser = argparse.ArgumentParser(description="Evaluate LDS score")
    parser.add_argument("model", help="Path to AirRep model state")
    parser.add_argument("pairs", help="Path to subset-loss pairs JSON")
    args = parser.parse_args()

    model = AirRep()
    model.model.load_state_dict(torch.load(args.model))
    pairs = load_pairs(args.pairs)
    score = evaluate_lds(model, pairs)
    print(f"LDS score: {score:.4f}")


if __name__ == "__main__":
    main()
