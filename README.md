# AirRep

AirRep is a scalable, representation-based data attribution model.
It introduces two key innovations: a trainable encoder optimized for attribution quality, and an attention-based pooling mechanism that enables accurate estimation of group-wise influence.

Paper: Enhancing Training Data Attribution with Representational Optimization ([arXiv:2505.18513](https://arxiv.org/pdf/2505.18513)).  
Author: Weiwei Sun, Haokun Liu, Nikhil Kandpal, Colin Raffel, Yiming Yang

## Quick Start

Use the FLAN-trained model ([https://huggingface.co/sunweiwei/AirRep-Flan-Small](https://huggingface.co/sunweiwei/AirRep-Flan-Small)) to encode training and test data and compute similarity scores.

```python
from airrep import AirRep

model = AirRep.from_pretrained("sunweiwei/AirRep-Flan-Small")

train_texts = [
    "Question: Classify the sentiment of 'The movie was wonderful and heartwarming.'\nAnswer: positive",
    "Question: Does the hypothesis entail the premise? Premise: 'A man is playing a guitar on stage.' Hypothesis: 'Someone is performing music.'\nAnswer: entailment",
]
query_texts = [
    "Question: Classify the sentiment of 'The service was awful and I won't return.'\nAnswer: negative"
]

# Embeddings and influence-like similarity score
train_emb = model.encode(train_texts, batch_size=128)
query_emb = model.encode(query_texts)
score = model.similarity(query_emb, train_emb, softmax=True)
print("Similarity score:", score)
```

Evaluate LDS score of `sunweiwei/AirRep-Flan-Small` on FLAN:

```bash
python scripts/04_evaluate.py \
  --model_path sunweiwei/AirRep-Flan-Small \
  --dataset sunweiwei/airrep-test \
  --benchmark flan
```

## Train AirRep

### 1) Create subset–dev pairs

```bash
python scripts/01_create_training_data.py \
  --data_path Muennighoff/flan \
  --output_path data/flan/train_index.jsonl \
  --dev_size 10000 \
  --train_size 100000 \
  --n_splits 100 \
  --subset_size 1000 \
  --n_subsets_per_split 100 \
  --seed 42
```

Output format:
```json
{
  "id": "0-0",
  "select": [1, 5, 10, ...],  // Indices of subset examples
  "dev": [0, 2, 3, ...],       // Indices of dev examples
  "dev_id": 0                   // Split ID
}
``````

### 2) SFT on subsets and collect dev losses

```bash
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python scripts/02_sft_subsets.py \
        --data_path Muennighoff/flan \
        --index_path data/flan/train_index.jsonl \
        --output_dir out/flan_sft \
        --model_name Qwen/Qwen2.5-0.5B \
        --start_idx $i --num_gpus 8 &
done
wait
```

Output format:
```json
{
  "pair_id": "0-0",
  "dev_id": 0,
  "losses": [2.1, 1.8, ...],  // Per-example losses on dev set
  "mean_loss": 1.95
}
```

### 3) Train the AirRep model

```bash
accelerate launch --num_processes 8 scripts/03_train_airrep.py \
    --data_path Muennighoff/flan \
    --index_path data/flan/train_index.jsonl \
    --loss_dir out/flan_sft \
    --save_path models/airrep-flan \
    --base_model thenlper/gte-small \
    --batch_size 1 \
    --epochs 50 \
    --lr 1e-4 \
    --topk 32 \
    --reference_size 1000
```

## Eval AirRep

Evaluate a trained model or the released model on the public evaluation set.

```bash
# Evaluate your trained model
python scripts/04_evaluate.py \
  --model_path models/airrep-flan \
  --dataset sunweiwei/airrep-test \
  --benchmark flan

# Or evaluate the released model
python scripts/04_evaluate.py \
  --model_path sunweiwei/AirRep-Flan-Small \
  --dataset sunweiwei/airrep-test \
  --benchmark flan
```

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{Sun2025AirRep,
  title={Enhancing Training Data Attribution with Representational Optimization},
  author={Weiwei Sun and Haokun Liu and Nikhil Kandpal and Colin Raffel and Yiming Yang},
  booktitle={NeurIPS},
  year={2025},
  url={https://arxiv.org/abs/2505.18513}
}
```
