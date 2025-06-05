# AirRep

AirRep provides attribution-friendly text embeddings trained for efficient training data analysis. The project accompanies our paper **"Enhancing Training Data Attribution with Representational Optimization"** ([arXiv:2505.18513](https://arxiv.org/pdf/2505.18513)).

## Installation

```bash
pip install -e .
```

## Usage

```python
from airrep import AirRep

texts = [
    "What is the capital of France?",
    "How tall is Mount Everest?",
]

model = AirRep()  # loads a small GTE based encoder by default
embeddings = model.encode(texts)
```

The `AirRep` class is a lightweight wrapper over `SentenceTransformer` and can be initialized with any compatible model name.

## Training

Generate training pairs and fit a regression head predicting language-model loss:

```bash
python scripts/train_airrep.py data.txt out_dir --samples 200 --subset_size 3
```

This creates `out_dir/pairs.json` and a trained model `out_dir/airrep.pt`.

## Evaluation

Evaluate LDS correlation on held-out pairs:

```bash
python scripts/evaluate_lds.py out_dir/airrep.pt test_pairs.json
```

## Inference

Compute influence scores for new data:

```bash
python scripts/inference.py out_dir/airrep.pt train.txt test.txt
```

## Citation

Please cite our work if you find this project useful:

```bibtex
@misc{sun2025enhancing,
  title        = {Enhancing Training Data Attribution with Representational Optimization},
  author       = {Weiwei Sun and Haokun Liu and Nikhil Kandpal and Colin Raffel and Yiming Yang},
  year         = {2025},
  eprint       = {2505.18513},
  archivePrefix= {arXiv},
  url          = {https://arxiv.org/abs/2505.18513},
  doi          = {10.48550/arXiv.2505.18513}
}
```
