import argparse

from airrep import AirRepModel, LDSEvaluator, load_pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluate LDS score")
    parser.add_argument("model", help="Path to AirRep regression head file")
    parser.add_argument("pairs", help="Path to subset-loss pairs JSON")
    args = parser.parse_args()

    model = AirRepModel.load(args.model)
    pairs = load_pairs(args.pairs)
    evaluator = LDSEvaluator(pairs)
    score = evaluator.eval(model)
    print(f"LDS score: {score:.4f}")


if __name__ == "__main__":
    main()
