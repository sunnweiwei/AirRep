import argparse
import torch
from airrep import AirRep, encode_texts, influence_scores


def main():
    parser = argparse.ArgumentParser(description="Encode texts and compute influence")
    parser.add_argument("model", help="Path to AirRep model state")
    parser.add_argument("train", help="Path to training text file")
    parser.add_argument("test", help="Path to test text file")
    args = parser.parse_args()

    model = AirRep()
    model.model.load_state_dict(torch.load(args.model))

    with open(args.train) as f:
        train_texts = [l.strip() for l in f if l.strip()]
    with open(args.test) as f:
        test_texts = [l.strip() for l in f if l.strip()]

    train_emb = encode_texts(model, train_texts)
    test_emb = encode_texts(model, test_texts)
    scores = influence_scores(train_emb, test_emb)
    print(scores)


if __name__ == "__main__":
    main()
