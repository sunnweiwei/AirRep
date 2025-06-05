# AirRep

AirRep provides attribution-friendly text embeddings trained for efficient training data analysis. The project accompanies our paper **"Enhancing Training Data Attribution with Representational Optimization"** ([arXiv:2505.18513](https://arxiv.org/pdf/2505.18513)).

## Installation

```bash
pip install -e .
```

The package depends on [SentenceTransformers](https://github.com/UKPLab/sentence-transformers).

## Quick start

```python
from airrep import AirRep

texts = [
    "What is the capital of France?",
    "How tall is Mount Everest?",
]

model = AirRep()  # loads a small GTE-based encoder by default
embeddings = model.encode(texts)
```

The `AirRep` class is a lightweight wrapper over `SentenceTransformer` and can be
initialized with any compatible model name.

## Training

Use `AirRepTrainer` to create subset–loss data and train a model:

```python
from airrep import AirRepTrainer

# dataset is a list of training texts
model, pairs = AirRepTrainer.fit(dataset, subset_size=3, samples=200)
```

The step-by-step API is also available:

```python
pairs = AirRepTrainer.generate_data(dataset)
model = AirRepTrainer.train_model(pairs)
```

## Evaluation

Create an `LDSEvaluator` with held-out pairs and measure correlation:

```python
from airrep import LDSEvaluator

evaluator = LDSEvaluator(test_pairs)
score = evaluator.eval(model)
```

## Inference

Encode texts and compute simple influence scores:

```python
from airrep import encode_texts, influence_scores

train_emb = encode_texts(model, train_texts)
test_emb = encode_texts(model, test_texts)
scores = influence_scores(train_emb, test_emb)
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
