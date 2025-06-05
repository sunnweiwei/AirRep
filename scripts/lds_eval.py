import torch
import torch.quantization as quant
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import numpy as np
from rbo import rbo
from collections import defaultdict
from tqdm import tqdm
import scipy.stats as stats
import faiss
import json
from file_io import *


def compute_tfidf(list1, list2):
    combined_list = list1 + list2
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_list)
    tfidf_matrix1 = tfidf_matrix[:len(list1)]
    tfidf_matrix2 = tfidf_matrix[len(list1):]
    similarity_matrix = cosine_similarity(tfidf_matrix1, tfidf_matrix2)
    return similarity_matrix


import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk import ngrams as get_ngrams
import hashlib
import multiprocessing as mp
from functools import partial
from typing import List
from tqdm.auto import tqdm

wpt = WordPunctTokenizer()


def hash_buckets(text: str, num_buckets: int) -> int:
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % num_buckets


def _process_text(args, n: int, num_buckets: int, tokenizer):
    text, = args
    counts = np.zeros(num_buckets, dtype=np.int32)
    words = tokenizer(text.lower())
    for w in words:
        counts[hash_buckets(w, num_buckets)] += 1
    for i in range(2, n + 1):
        for ng in get_ngrams(words, i):
            counts[hash_buckets(' '.join(ng), num_buckets)] += 1
    return counts


def dsir(test_data: List[str], train_data: List[str],
         n: int = 2, num_buckets: int = 10000,
         laplace_smoothing: float = 0.0, n_jobs: int = -1) -> np.ndarray:
    # Parallel processing setup
    n_jobs = 64 if n_jobs == -1 else n_jobs
    process_fn = partial(_process_text, n=n, num_buckets=num_buckets, tokenizer=wpt.tokenize)

    # Create argument lists with progress tracking
    train_args = [(t,) for t in train_data]
    test_args = [(t,) for t in test_data]

    # Parallel feature extraction with progress bars
    with mp.Pool(n_jobs) as pool:
        # Process training data with progress bar
        train_features = list(tqdm(
            pool.imap(process_fn, train_args, chunksize=100),
            total=len(train_data),
            desc="Processing training data",
            unit="doc"
        ))

        # Process test data with progress bar
        test_features = list(tqdm(
            pool.imap(process_fn, test_args, chunksize=100),
            total=len(test_data),
            desc="Processing test data",
            unit="doc"
        ))

    # Convert to numpy arrays
    train_matrix = np.array(train_features, dtype=np.float32)
    test_matrix = np.array(test_features, dtype=np.float32)

    # Compute probabilities with vectorized operations
    with tqdm(total=3, desc="Computing probabilities") as pbar:
        raw_probs = (train_matrix.sum(axis=0) / train_matrix.sum()).astype(np.float32)
        pbar.update(1)
        test_matrix += laplace_smoothing
        target_probs = test_matrix / test_matrix.sum(axis=1, keepdims=True)
        pbar.update(1)

        # Compute log difference and final matrix
        log_diff = np.log(target_probs + 1e-8) - np.log(raw_probs + 1e-8)
        result = log_diff @ train_matrix.T.astype(np.float32)
        pbar.update(1)

    return result


def compute_embed(list1, list2):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import dot_score
    model_name = 'Alibaba-NLP/gte-base-en-v1.5'
    # model_kwargs = {"torch_dtype": torch.float16, 'trust_remote_code=True': True}
    # model_kwargs = {'trust_remote_code=True': True}
    model = SentenceTransformer(model_name, trust_remote_code=True)
    combined_list = list1 + list2
    embed_matrix = model.encode(combined_list, show_progress_bar=True)
    embed_matrix1 = embed_matrix[:len(list1)]
    embed_matrix2 = embed_matrix[len(list1):]
    # similarity_matrix = dot_score(embed_matrix1, embed_matrix2)
    embed = embed_matrix1
    mem = embed_matrix2
    torch.save(mem, 'out/gte/gte-alpaca.pt')
    all_scores = np.zeros(shape=(len(embed), len(mem)))
    bs = 1024

    embed = torch.tensor(embed).cuda().float()
    for i in tqdm(range(0, len(mem), bs), total=len(mem) // bs):
        batch = mem[i:i + bs]
        batch = torch.tensor(batch, device='cuda').float()
        score = torch.mm(embed, batch.T).cpu().float().numpy()
        all_scores[:, i:i + bs] = score
    return all_scores


def apply_pq(pq, x):
    code = pq.encode(x)  # compression
    return index.decode(code)  # reconstruction
    # codes = pq.compute_codes(x)
    # return pq.decode(codes)


def match(embed, train_mem):
    all_scores = np.zeros(shape=(len(embed), len(train_mem)))
    bs = 2048
    embed = embed.cuda().float()
    for i in tqdm(range(0, len(train_mem), bs), total=len(train_mem) // bs):
        batch = train_mem[i:i + bs]
        batch = torch.tensor(batch, device='cuda').float()
        score = torch.mm(embed, batch.T).cpu().float().numpy()
        all_scores[:, i:i + bs] = score
    return all_scores


def max_sum(a, k=10):
    c = sorted(a, reverse=True)
    return sum(c[:k])


def power_sum(a, k=9):
    return (np.array(a) ** k).sum()


def softmax(a, k=10):
    # c = sorted(a, reverse=True)
    # a = c[:k]
    a = torch.tensor(a)
    attn = a.softmax(dim=-1)
    a = attn * a
    return a.sum().item()


def main(data_name='tulu', sum_func=sum, epoch=0):
    print('#' * 50)
    print(data_name)
    print(sum_func)
    print('#' * 50)

    
    # embed = torch.load(f"out/Qwen2.5-3B-Less/dim768-test/all_orig.pt")
    # train_mem = torch.load(f"out/Qwen2.5-3B-Less/dim768-train/all_orig.pt")
    # score1 = match(embed, train_mem)
    # print(score1.shape)

    save_path = f'models/flan-gte-small-pool-3'
    epoch = '8'
    if not os.path.exists(f'{save_path}/embeddings/flan-embeddings-{epoch}.pt'):
        return
    all_embed = torch.load(f'{save_path}/embeddings/flan-embeddings-{epoch}.pt')
    train_id = all_embed[:-6520]
    test_embed = all_embed[-6520:]
    score1 = match(test_embed, train_id)

    score2 = score1

    num_run = 100
    loss = [json.load(open(f'out/flan_lite_lds/loss-{j}.json')) for j in range(num_run)]
    # loss = [json.load(open(f'out/flan-3B_lds/loss-{j}.json')) for j in range(num_run)]
    # loss = [json.load(open(f'out/flan_lds/loss-{j}.json')) for j in range(num_run)]
    loss = torch.tensor(loss)
    loss = loss.mean(dim=0, keepdim=True) - loss
    loss = loss.tolist()

    # index = read_jsonl('data/flan/train_lite_index.jsonl')
    index = read_jsonl('data/flan/train_lite_index.jsonl')
    index = [x['select'] for x in index]
    # print(score1.shape)

    train_type = [x['task'] for x in read_jsonl('data/flan/train_lite.jsonl')]
    test_type = [x['task'] for x in read_jsonl('data/flan/test_lite.jsonl')]

    metric = defaultdict(list)

    all_model = []
    # 6520
    for i in tqdm(range(6520), total=6520):
        model = []
        label = []
        for j in range(num_run):
            # model.append(score1[j][i])
            set_score = score2[i, index[j]]
            model.append(sum_func(set_score))
            label.append(loss[j][i])
        correlation, p_value = stats.spearmanr(model, label)
        correlation = 0 if np.isnan(correlation) else correlation
        metric['Spearman'].append(float(correlation))
        all_model.append(model)
        metric['GroupAcc'].append(int(np.argmax(model) == np.argmax(label)))

        rank1 = np.argsort(-score1[i]).tolist()
        pred_type = [train_type[t] for t in rank1[:100]]
        metric['Type@1'].append(int(test_type[i] in pred_type[:1]))
        metric['Type@10'].append(int(test_type[i] in pred_type[:10]))

    all_model = np.array(all_model)
    model = []
    label = []
    for j in range(num_run):
        # model.append(score2[:, index[j]].sum())
        model.append(all_model[:, j].sum())
        # model.append(sum([sum([score2[i][t] for t in index[j]]) for i in range(len(score1))]))
        label.append(-sum(loss[j]))
    # print(label)
    # print(model)
    correlation, p_value = stats.spearmanr(model, label)
    metric['AllSpearman'].append(float(correlation))

    rd = lambda x: f"{x:.2f}"

    for k, v in metric.items():
        print(k, rd(sum(v) / len(v) * 100))


def master():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int, default=0)
    args = parser.parse_args()
    start = args.i
    # main('flan_lite', softmax, epoch=start)
    main('flan_lite', sum, epoch=start)


master()

# main('alpaca', sum)
# main('alpaca', max_sum)
# main('alpaca', softmax)

# main('tulu', sum)
# main('tulu', max_sum)
# main('tulu', softmax)

# main('safe', sum)
# main('safe', max_sum)
# main('safe', softmax)

