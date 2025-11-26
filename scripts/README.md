# AirRep Training Pipeline

This directory contains the three-stage training pipeline for AirRep models.

## Overview

AirRep training consists of three stages:

1. **Data Sampling**: Create cross-validation splits and sample subset-dev pairs
2. **SFT Training**: Fine-tune models on each subset and evaluate on dev sets
3. **AirRep Training**: Train the AirRep influence prediction model

## Stage 1: Create Training Data

Creates subset-dev pairs for training.

```bash
python 01_create_training_data.py \
    --data_path Muennighoff/flan \
    --output_path data/flan/train_index.jsonl \
    --dev_size 10000 \
    --train_size 100000 \
    --n_splits 100 \
    --subset_size 1000 \
    --n_subsets_per_split 100
```

**Parameters:**
- `--data_path`: Path to training data (jsonl, json, or HuggingFace dataset)
- `--output_path`: Path to save subset-dev pairs
- `--dev_size`: Size of development set (default: 10000)
- `--train_size`: Size of training set after dev split (default: 100000)
- `--n_splits`: Number of cross-validation splits (default: 100)
- `--subset_size`: Size of each training subset (default: 1000)
- `--n_subsets_per_split`: Number of subsets per split (default: 100)

**Output:** A jsonl file where each line contains:
```json
{
  "id": "0-0",
  "select": [1, 5, 10, ...],  // Indices of subset examples
  "dev": [0, 2, 3, ...],       // Indices of dev examples
  "dev_id": 0                   // Split ID
}
```

## Stage 2: SFT on Subsets

Fine-tune models on each subset and evaluate on corresponding dev sets.

```bash
# Single GPU
python 02_sft_subsets.py \
    --data_path Muennighoff/flan \
    --index_path data/flan/train_index.jsonl \
    --output_dir out/flan_sft \
    --model_name Qwen/Qwen2.5-0.5B \
    --batch_size 16 \
    --gas 1 \
    --epochs 2 \
    --start_idx 0 \
    --num_gpus 1

# Multi-GPU parallel (8 GPUs)
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python 02_sft_subsets.py \
        --data_path Muennighoff/flan \
        --index_path data/flan/train_index.jsonl \
        --output_dir out/flan_sft \
        --model_name Qwen/Qwen2.5-0.5B \
        --start_idx $i --num_gpus 8 &
done
wait
```

**Parameters:**
- `--data_path`: Path to training data
- `--index_path`: Path to subset-dev pairs from stage 1
- `--output_dir`: Directory to save losses
- `--model_name`: Base model for SFT (default: Qwen/Qwen2.5-0.5B)
- `--batch_size`: Training batch size (default: 16)
- `--gas`: Gradient accumulation steps (default: 1)
- `--epochs`: Number of epochs (default: 2)
- `--start_idx`: Start index for parallel processing
- `--num_gpus`: Number of GPUs for parallel processing

**Output:** For each pair, saves `loss-{pair_id}.json`:
```json
{
  "pair_id": "0-0",
  "dev_id": 0,
  "losses": [2.1, 1.8, ...],  // Per-example losses on dev set
  "mean_loss": 1.95
}
```

## Stage 3: Train AirRep Model

Train the AirRep influence prediction model (requires 8 GPUs).

```bash
accelerate launch --num_processes 8 03_train_airrep.py \
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

**Parameters:**
- `--data_path`: Path to training data
- `--index_path`: Path to subset-dev pairs from stage 1
- `--loss_dir`: Directory with losses from stage 2
- `--save_path`: Path to save trained AirRep model
- `--base_model`: Base encoder model (default: thenlper/gte-small)
- `--batch_size`: Training batch size (default: 1)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--topk`: Number of top subsets to use (default: 32)
- `--reference_size`: Number of dev examples to use (default: 1000)

**Output:** Trained AirRep model saved to `save_path`

## Stage 4: Evaluate Model

Evaluate the trained AirRep model on test datasets.

```bash
python 04_evaluate.py \
    --model_path models/airrep-flan \
    --dataset sunweiwei/airrep-test \
    --benchmark flan \
    --batch_size 128 \
    --output_dir ./eval_results
```

**Parameters:**
- `--model_path`: Path to trained AirRep model
- `--dataset`: HuggingFace dataset for evaluation (default: sunweiwei/airrep-test)
- `--benchmark`: Benchmark name (default: flan)
- `--batch_size`: Batch size for encoding (default: 128)
- `--output_dir`: Directory to save embeddings and results (default: .)

**Output:**
- Embeddings saved as `{benchmark}_train_emb.npy` and `{benchmark}_test_emb.npy`
- Results saved as `{benchmark}_results.json`
- Prints LDS Spearman correlation (expected: ~0.21 for flan)

## Usage After Training

```python
from airrep import AirRep

# Load trained model
model = AirRep.from_pretrained("models/airrep-flan")

# Encode texts
train_embed = model.encode(train_texts, batch_size=128)
test_embed = model.encode(test_texts, batch_size=128)

# Compute influence scores
scores = model.similarity(test_embed, train_embed, softmax=True)
```

## Complete Example

```bash
# Stage 1: Create subset-dev pairs
python 01_create_training_data.py \
    --data_path Muennighoff/flan \
    --output_path data/flan/train_index.jsonl

# Stage 2: SFT on subsets (8 GPUs in parallel)
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python 02_sft_subsets.py \
        --data_path Muennighoff/flan \
        --index_path data/flan/train_index.jsonl \
        --output_dir out/flan_sft \
        --start_idx $i --num_gpus 8 &
done
wait

# Stage 3: Train AirRep model
accelerate launch --num_processes 8 03_train_airrep.py \
    --data_path Muennighoff/flan \
    --index_path data/flan/train_index.jsonl \
    --loss_dir out/flan_sft \
    --save_path models/airrep-flan

# Stage 4: Evaluate model
python 04_evaluate.py \
    --model_path models/airrep-flan \
    --benchmark flan \
    --output_dir ./eval_results
```

