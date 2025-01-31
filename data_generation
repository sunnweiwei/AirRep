import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from collections import defaultdict
import shutil
import numpy as np
from vllm import LLM, SamplingParams
from file_io import *

# Enable TF32 if possible
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

import os



def built_example(messages, tokenizer):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    q_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    labels = [-100] * len(q_ids) + ids[len(q_ids):]
    input_ids = ids
    return input_ids, labels


class TrainData(Dataset):
    def __init__(self, data, tokenizer, max_len=800):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        item = self.data[index]
        if 'data' in item:
            prompt = item['data'][0]
            response = item['data'][1]

        if 'input' in item:
            prompt = item['instruction']

        for op in ['inputs', 'prompt', 'instruction']:
            if op in item:
                prompt = item[op]
                break

        for op in ['response', 'targets', 'output']:
            if op in item:
                response = item[op]
                break

        prompt = " ".join(prompt.split()[:100])
        prompt = 'Question: ' + prompt + '\nAnswer:'
        msg = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
        input_ids, labels = built_example(msg, self.tokenizer)
        return torch.tensor(input_ids[:self.max_len]), torch.tensor(labels[:self.max_len])

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        features = {
            "input_ids": input_ids,
            "labels": labels,
        }
        return features


class TestData(TrainData):
    def __getitem__(self, index):
        item = self.data[index]
        if 'data' in item:
            prompt = item['data'][0]
            response = item['data'][1]

        if 'input' in item:
            prompt = item['instruction']

        for op in ['inputs', 'prompt', 'instruction']:
            if op in item:
                prompt = item[op]
                break

        for op in ['response', 'targets', 'output']:
            if op in item:
                response = item[op]
                break

        prompt = " ".join(prompt.split()[:100])
        prompt = 'Question: ' + prompt + '\nAnswer:'
        msg = [{'role': 'user', 'content': prompt}]
        q_ids = self.tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
        return torch.tensor(q_ids), torch.tensor(q_ids)

    def collate_fn(self, batch):
        input_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        features = {
            "input_ids": input_ids,
            "labels": labels,
        }
        return features


def main(index=0, data_name='tulu', data=None, model=None, tokenizer=None, save_model=False, save_path=None,
         model_name=None, gas=1):
    accelerator = Accelerator(gradient_accumulation_steps=gas)
    # model_name = 'models/Qwen2.5-0.5B-Instruct'
    model_name = model_name or 'models/Qwen2.5-1.5B'
    save_path = save_path or f'models/Qwen2.5-0.5B-Flan-Tmp'
    batch_size = 16 // gas
    epochs = 2

    # prepare model, tokenizer, and data loader
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(len(data))

    dataset = TrainData(data, tokenizer, max_len=800)
    print(len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    # if accelerator.is_main_process:
    for epoch in range(epochs):
        accelerator.print(f'Training {epoch}')
        # accelerator.wait_for_everyone()
        # model.train()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=not accelerator.is_main_process)
        losses = []
        for batch in tk0:
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                if torch.isnan(loss):
                    loss = torch.tensor(0.0, requires_grad=True).to(loss.device)
                    print('NaN Error!')
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            losses.append(loss.item())
            tk0.set_postfix(loss=sum(losses[-500:]) / len(losses[-500:]))
    if save_model:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f'{save_path}',
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(f'{save_path}')
        accelerator.wait_for_everyone()
    return accelerator.unwrap_model(model)


@torch.no_grad()
def eval_generation(data, seed=0, data_name='tulu', model=None, tokenizer=None):
    import numpy as np
    # index = 0
    # model_name = f'models/TinyLlama-1.1B-Select-{index}'
    model_name = 'models/Qwen2.5-0.5B-Flan-Tmp'
    # name = data_name[0].upper() + data_name[1:]
    # model_name = f'models/TinyLlama-1.1B-{name}'
    if model is None:
        gpu_id = seed % 8
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        model = model.to(f'cuda:{gpu_id}')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # dataset = TrainData([json.loads(x) for x in open('data/alpaca-output.jsonl')], tokenizer)
    # dataset = TrainData(json.load(open('data/safe-test.json')), tokenizer)
    if data is None:
        data = read_jsonl('data/flan/test_100.jsonl')
    # dataset = TestData(data, tokenizer)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=16, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4
    # )
    model.eval()
    # os.system('sh kill_gpu.sh')
    losses = []
    all_pred = []
    all_label = []
    output = []
    for item in tqdm(data):
        for op in ['inputs', 'prompt', 'instruction']:
            if op in item:
                prompt = 'Question: ' + " ".join(item[op].split()[:128]) + '\nAnswer:'
                break

        for op in ['response', 'targets', 'output']:
            if op in item:
                response = item[op]
                break
        msg = [{'role': 'user', 'content': prompt}]
        q_ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([q_ids], device=model.device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, max_new_tokens=64, eos_token_id=151645, pad_token_id=151645)
            out = out[0]
            out = out[len(q_ids):]
            text = tokenizer.decode(out)
            text = text.replace('<|im_end|>', '')
            all_pred.append(text)
        # all_label.append(response)
        # print(item)
        # print(text)
        # print(eval_f1(all_pred, all_label))
        # item['prediction'] = text
        # output.append(item)
    all_label = [x['targets'] for x in data]
    return all_pred, all_label


def eval_generation_hub():
    torch.multiprocessing.set_start_method('spawn')
    data = read_jsonl('data/flan/test_100.jsonl')
    np.random.shuffle(data)
    output = mp(eval_generation, data, processes=24)
    all_pred = [x['prediction'] for x in output]
    all_label = [x['targets'] for x in output]
    print(eval_f1(all_pred, all_label))

    results = defaultdict(list)
    for item in output:
        results[item['task']].append(item)
    new_data = []
    for name, test in results.items():
        all_pred = [x['prediction'] for x in test]
        all_label = [x['targets'] for x in test]
        print(name, len(test), eval_f1(all_pred, all_label))
    write_jsonl(output, 'out/flan_out/flan_test100.jsonl')


@torch.no_grad()
def eval_ppl(index=0, data_name='flan', data=None, model=None, tokenizer=None):
    import numpy as np
    # index = 0
    accelerator = Accelerator()
    # model_name = f'models/TinyLlama-1.1B-Select-{index}'
    # model_name = 'models/Qwen2.5-0.5B-Instruct'
    model_name = 'models/Qwen2.5-0.5B-Flan'
    # name = data_name[0].upper() + data_name[1:]
    # model_name = f'models/TinyLlama-1.1B-{name}'
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if data is not None:
        dataset = TrainData(data, tokenizer, max_len=800)
    elif data_name == 'tulu':
        dataset = TrainData(json.load(open('data/tulu-test.json')), tokenizer)
    elif data_name == 'alpaca':
        dataset = TrainData([json.loads(x) for x in open('data/alpaca-output.jsonl')], tokenizer)
    elif data_name == 'safe':
        dataset = TrainData(json.load(open('data/safe-test.json')), tokenizer)
    elif data_name == 'medqa':
        dataset = TrainData(json.load(open('data/medqa/test.json')), tokenizer)
    elif data_name == 'flan':
        dataset = TrainData(read_jsonl('data/flan/test_100.jsonl'), tokenizer, max_len=2048)
    # dataset = TrainData([json.loads(x) for x in open('data/alpaca-output.jsonl')], tokenizer)
    # dataset = TrainData(json.load(open('data/safe-test.json')), tokenizer)
    # dataset = TrainData(json.load(open('data/tulu-test.json')), tokenizer)

    print(len(dataset))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4
    )
    model.eval()
    losses = []
    for batch in tqdm(data_loader):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            # out = model(**batch)
            lm_logits = model(batch['input_ids']).logits
            bs = lm_logits.size(0)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            loss = loss.view(bs, -1)
            seq_len = batch['labels'].ne(-100).sum(dim=-1)
            loss = loss.sum(dim=-1) / seq_len
            loss[seq_len == 0] = 10
        # loss = out.loss
        loss = loss.cpu().tolist()
        losses.extend(loss)
    print(sum(losses) / len(losses))
    variance = np.var(losses)
    print(variance)
    # json.dump(losses, open(f'out/tulu_lds/loss-{index}.json', 'w'))
    if accelerator.is_main_process:
        os.makedirs(f'out/{data_name}_lds', exist_ok=True)
        json.dump(losses, open(f'out/{data_name}_lds/loss-{index}.json', 'w'))
    accelerator.wait_for_everyone()


def data_gen():
    import gc
    import argparse
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=int, default=0)
    args = parser.parse_args()
    start = args.i

    model_name = 'models/Qwen2.5-3B'

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    raw_state = raw_model.state_dict()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    name = 'AirRep'

    model_version = '3B-'

    data_num = 1000
    version = 1

    data_name = f'flan_select/{name}/{model_version}{data_num}-{version}'

    print(data_name, f'data/flan_select/{name}.jsonl')

    data = read_jsonl('data/flan/train_lite.jsonl')
    test_data = read_jsonl('data/flan/test_100.jsonl')
    index = read_jsonl(f'data/flan_select/{name}.jsonl')

    for i in range(start, 100, 8):
        # if os.path.exists(f'out/{data_name}_lds/loss-{i}.json'):
        #     continue
        model.load_state_dict(raw_state)
        print('RUN', i)
        data_sub = [data[x] for x in index[i]['select']]
        # data_sub = [data[x] for x in index[i]]
        data_sub = data_sub[:data_num]
        model = main(i, model=model, data=data_sub, tokenizer=tokenizer, gas=4)
        torch.cuda.empty_cache()

        # test_data_sub = test_data
        # test_data_sub = [test_data[x] for x in index[i]['dev']]
        # eval_ppl(i, data_name, model=model, data=test_data_sub, tokenizer=tokenizer)

        test_data_sub = [x for x in test_data if x['task'] == index[i]['task']]
        eval_ppl(i, data_name, model=model, data=test_data_sub, tokenizer=tokenizer)
        all_pred, all_label = eval_generation(test_data_sub, seed=start, data_name=data_name, model=model,
                                              tokenizer=tokenizer)
        f1 = eval_f1(all_pred, all_label)
        print(index[i]['task'], f1)
        write_json(
            {"data_name": data_name, "task": index[i]['task'], "f1": f1},
            f'out/{data_name}_lds/f1-{i}.json')



if __name__ == "__main__":
    data_gen()



