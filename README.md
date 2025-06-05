# AirRep

AirRep provides attribution-friendly text embeddings trained for efficient training data analysis. The project accompanies our paper **"Enhancing Training Data Attribution with Representational Optimization"** ([arXiv:2505.18513](https://arxiv.org/pdf/2505.18513)).

## Installation

```bash
pip install -e .
```

The package depends on [SentenceTransformers](https://github.com/UKPLab/sentence-transformers).

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

## Training & Evaluation

Additional scripts for data generation, model training and evaluation are available in the `scripts/` directory:

- `data_generation.py` – create training pairs
- `train.py` – train the AirRep encoder
- `lds_eval.py` – evaluate training data attribution
- `data_selection.py` – helper utilities for dataset curation

Run them with `python scripts/<name>.py`.

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
