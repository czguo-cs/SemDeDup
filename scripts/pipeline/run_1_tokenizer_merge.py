import numpy as np
from multiprocessing import Pool,cpu_count
from torch.utils.data import DataLoader 
import sys
import os
sys.path.append("/home/guochuanzhe/data-process/SemDeDup")
# from compute_pretrained_embeddings import get_embeddings,get_nl_embeddings
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import Dataset
import json
import torch
from tqdm.auto import tqdm
from torch.nn.functional import normalize
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import fire
import pickle


def process_chunk(chunk_data):
    tokenizer, max_length = chunk_data[0], chunk_data[1]
    results = []

    for line in tqdm(chunk_data[2:], desc="tokenizer and chunk..."):
        data_line = json.loads(line)
        text = data_line['content'] if 'content' in data_line else data_line['text']
        data = tokenizer(text, truncation=False, return_tensors="pt", padding=False)
        input_ids = data['input_ids'].squeeze(0)

        # 分段处理
        input_ids_chunks = []
        attention_mask_chunks = []
        for i in range(0, len(input_ids), max_length):
            chunk = input_ids[i:i + max_length]
            input_ids_chunks.append(chunk)
            attention_mask_chunks.append(torch.ones(len(chunk), dtype=torch.long))

        # 只对最后一个向量进行填充
        last_chunk = input_ids_chunks[-1]
        if len(last_chunk) < max_length:
            padding_size = max_length - len(last_chunk)
            padded_last_chunk = torch.cat([last_chunk, torch.tensor([1]*padding_size)])
            input_ids_chunks[-1] = padded_last_chunk

            attention_mask_chunks[-1] = torch.cat([attention_mask_chunks[-1], torch.tensor([0]*padding_size)])

        # 将列表转换为张量
        input_ids_chunks = torch.stack(input_ids_chunks)
        attention_mask_chunks = torch.stack(attention_mask_chunks)

        results.append((input_ids_chunks, attention_mask_chunks))

    return results

class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.input_ids = []
        self.attention_mask = []
        self.index = []
        self.index_num = []

        # 读取文件并分割成多个区域
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        num_process=32
        chunk_size = len(lines) // num_process + 1
        chunks = [(tokenizer, max_length) + tuple(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]

        # 使用多进程处理
        with Pool(num_process) as p:
            chunk_results = p.map(process_chunk, chunks)
        count=0
        for chunk in chunk_results:
            for input_ids_chunks, attention_mask_chunks in chunk:
                self.index_num.append(len(input_ids_chunks))
                for input_ids_chunk, attention_mask_chunk in zip(input_ids_chunks, attention_mask_chunks):
                    self.input_ids.append(input_ids_chunk)
                    self.attention_mask.append(attention_mask_chunk)
                    self.index.append(count)
                count+=1

    def __len__(self):
        return len(self.index_num)
    
    def get_index_num(self):
        return self.index_num
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.index[idx]
    
def main(
    model_path:str = "/home/guochuanzhe/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    model_name:str = "opt-125M",
    dataset_name:str = "python",
    split:str = "train",
    type:str = "front",
    emb_size:int = 768, 
):
    # 基础元素初始化
    print(f"Start running model {model_name} for dataset {dataset_name}-{split} in the way of {type}")
    dataset_loc = f"/home/guochuanzhe/data-process/SemDeDup/memory/dataset/{dataset_name}/{type}/{model_name}/jsonl_dataset.pkl"
    os.makedirs(os.path.dirname(dataset_loc), exist_ok=True)
    path_data = f"/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-{dataset_name}-{split}.jsonl"
    # tokenizer and model 
    print(f"start load model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        truncation=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="</s>"))
    print(f"tokenizer loaded successfully")
    dataset=JsonlDataset(path_data,tokenizer)
    # 将对象保存到文件
    with open(dataset_loc, 'wb') as file:
        pickle.dump(dataset, file)
    print(f"dataset saved successfully")
    

if __name__  == '__main__':
    fire.Fire(main)

