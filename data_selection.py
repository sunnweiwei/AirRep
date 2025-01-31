import numpy as np
from tqdm import tqdm
import json
import os
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import torch
import zlib
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


def tfidf_score(data_name):
    if data_name == 'alpaca':
        train = [x['output'] for x in load_dataset('yahma/alpaca-cleaned')['train']]
        test = [json.loads(x)['output'] for x in open('data/alpaca-output.jsonl')]
        all_scores = compute_tfidf(test, train)
        all_scores = all_scores ** 10
        all_scores[np.isnan(all_scores)] = 0
    elif data_name == 'safe':
        test = [x['response'] for x in json.load(open('data/safe-test.json'))]
        train = [str(x['response']) for x in load_dataset('PKU-Alignment/PKU-SafeRLHF-QA')['train']]
        all_scores = compute_tfidf(test, train)
    elif data_name == 'tulu':
        train = [x['output'] for x in json.load(open('data/tulu-train-lite.json'))]
        test = [x['output'] for x in json.load(open('data/tulu-test.json'))]
        all_scores = compute_tfidf(test, train)
    return all_scores


def zip_fit(string1, string2):
    size_string1 = sys.getsizeof(string1)
    size_string2 = sys.getsizeof(string2)
    concatenated_string = string1 + string2
    compressed_data = zlib.compress(concatenated_string.encode())
    compressed_size = sys.getsizeof(compressed_data)
    return (compressed_size - min(size_string1, size_string2)) / max(size_string1, size_string2)


def zip_fit_score(data_name):
    if data_name == 'alpaca':
        train = [x['output'] for x in load_dataset('yahma/alpaca-cleaned')['train']]
        test = [json.loads(x)['output'] for x in open('data/alpaca-output.jsonl')]
    elif data_name == 'safe':
        test = [x['response'] for x in json.load(open('data/safe-test.json'))]
        train = [str(x['response']) for x in load_dataset('PKU-Alignment/PKU-SafeRLHF-QA')['train']]
    elif data_name == 'tulu':
        train = [x['output'] for x in json.load(open('data/tulu-train-lite.json'))]
        test = [x['output'] for x in json.load(open('data/tulu-test.json'))]

    all_scores = np.zeros((len(test), len(train)))
    for i in tqdm(range(len(test)), total=len(test)):
        for j in range(len(train)):
            all_scores[i, j] = zip_fit(test[i], train[j])

    return all_scores


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


def selection():
    for data_name in ['flan']:
        test_task = [x['task'] for x in read_jsonl('data/flan/test_lite.jsonl')]
        train_set = read_jsonl('data/flan/train_lite.jsonl')
        print(len(train_set))

        name = 'AirRep'

        save_path = f'models/flan-gte-small-pool-3'
        epoch = '8'
        if not os.path.exists(f'{save_path}/embeddings/flan-embeddings-{epoch}.pt'):
            return
        all_embed = torch.load(f'{save_path}/embeddings/flan-embeddings-{epoch}.pt')
        train_id = all_embed[:-6520]
        test_embed = all_embed[-6520:]
        score1 = match(test_embed, train_id)

        train_rank = []
        print(score1.shape)
        for i in tqdm(range(len(score1)), total=len(score1)):
            rank1 = np.argsort(-score1[i]).tolist()
            train_rank.append(rank1)
        print(len(train_rank), len(train_rank[0]))

        all_task = list(set(test_task))
        all_task.sort()

        all_index = []
        max_size = 10000
        for task in all_task:
            print(task)
            index = []
            for i in range(1000):
                for j in range(len(train_rank)):
                    if test_task[j] != task:
                        continue
                    if train_rank[j][i] not in index:
                        index.append(train_rank[j][i])
                    if len(index) >= max_size:
                        break
                if len(index) >= max_size:
                    break
            print(len(index))
            print()
            # index = [i for i in range(len(train_set))]
            # np.random.shuffle(index)
            # index = index[:max_size]
            all_index.append({'task': task, 'select': index})

        # index = [i for i in range(len(train_set))]
        # np.random.shuffle(index)
        # index = index[:10000]
        # subset = [train_set[i] for i in index]
        # print(len(index))
        os.makedirs(f'data/{data_name}_select', exist_ok=True)
        # write_jsonl(subset, f'data/{data_name}_select/{name}.jsonl')
        print(f'data/{data_name}_select/{name}.jsonl')
        write_jsonl(all_index, f'data/{data_name}_select/{name}.jsonl')


def eval():
    name = 'AirRep'

    data_num = 1000
    version = 1
    model_version = '3B-'
    # model_version = ''

    data_name = f'flan_select/{name}/{model_version}{data_num}-{version}'
    print(data_name)

    all_f1 = []
    all_loss = []
    data = []
    for i in range(66):
        if not os.path.exists(f'out/{data_name}_lds/f1-{i}.json'):
            continue
        f1 = read_json(f'out/{data_name}_lds/f1-{i}.json')
        loss = read_json(f'out/{data_name}_lds/loss-{i}.json')
        task = f1['task']
        data.append([task, f1['f1']['F1'], sum(loss) / len(loss)])

    data.sort(key=lambda x: x[0])
    for line in data:
        all_f1.append(line[1])
        all_loss.append(line[2])

    from tabulate import tabulate
    print(tabulate(data, headers=['model', 'f1', 'loss'], tablefmt="grid"))
    print(name)
    print(len(all_f1))
    print(sum(all_f1) / len(all_f1), sum(all_loss) / len(all_loss))


if __name__ == '__main__':
    eval()
    # selection()

