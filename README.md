# AirRep: Enhancing Training Data Attribution with Representational Optimization


Training data attribution (TDA) methods aim to measure how training data impacts a model's predictions. 
While gradient-based attribution methods, such as influence functions, offer theoretical grounding, their computational costs make them impractical for large-scale applications. 
Representation-based approaches are far more scalable, but typically rely on heuristic embeddings that are not optimized for attribution, limiting their fidelity.
To address these challenges, we propose AirRep, 
a scalable, representation-based approach that closes this gap by learning task-specific and model-aligned representations optimized explicitly for TDA.
AirRep introduces two key innovations: a trainable encoder tuned for attribution quality, and an attention-based pooling mechanism that enables accurate estimation of group-wise influence.
We train AirRep using a ranking objective over automatically constructed training subsets labeled by their empirical effect on target predictions.
Experiments on instruction-tuned LLMs
demonstrate that AirRep achieves performance on par with state-of-the-art gradient-based approaches while being nearly two orders of magnitude more efficient at inference time.
Further analysis highlights its robustness 
and generalization across tasks and models.

Paper: [https://arxiv.org/pdf/2505.18513](https://arxiv.org/pdf/2505.18513)

Note: This is a code draft; a more formal version will be released later.

# Model Inference

```
from airrep import AirRep

data = [
  "xxxx",
  "yyy"
]

model = AirRep('xxx/airrep-small-flan')
embeddings = model.encode(data)
```

# Model Training


Data Generation: `python data_generation.py`

Model Training: `python train.py`

LDS Evaluation: `python lds_eval.py`

Data Selection Evaluation: `python data_selection.py`

# Cite

```
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

