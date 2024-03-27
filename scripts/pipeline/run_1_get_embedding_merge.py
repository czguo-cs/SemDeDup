import numpy as np
import multiprocessing
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
from datasets import load_from_disk

def extract_sequence_lengths(input_list):
    result = []
    current_length = 1
    for i in range(1, len(input_list)):
        if input_list[i] == input_list[i-1]:
            current_length += 1
        else:
            result.append(current_length)
            current_length = 1
    result.append(current_length)
    return result


def main(
    model_path:str = "/home/guochuanzhe/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    model_name:str = "opt-125M",
    dataset_name:str = "python",
    split:str = "train",
    type:str = "front",
    emb_size:int = 768, 
):
    # 并行初始化
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    # 基础元素初始化
    print(f"Start running model {model_name} for dataset {dataset_name}-{split} in the way of {type} on rank {rank}.")
    emb_memory_loc = f"/home/guochuanzhe/data-process/SemDeDup/memory/embedding/{dataset_name}/{type}/{model_name}/emb_memory_loc.npy"
    dataset_path=f"/home/guochuanzhe/data-process/SemDeDup/data/{dataset_name}/encoded-starcoder-{dataset_name}-{split}/{model_name}"
    # tokenizer and model 
    print(f"start load model and tokenizer on rank {rank}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        truncation=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="</s>"))
    # 与原来相比，opt-125M由f16改为bf16
    if model_name == "opt-125M":
        torch_dtype=torch.float16
        print("model load in torch.float16")
    else:
        torch_dtype=torch.bfloat16
        print("model load in torch.bfloat16")
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2"
    )
    print(f"model and tokenizer loaded successfully on rank {rank}")
    dataset=load_from_disk(dataset_path) 
    index_num = extract_sequence_lengths(dataset['idx'])
    dataset_size = len(index_num)  
    print(f"dataset_size:{dataset_size}")                                               # starcoder
    # 创建父目录（如果不存在）
    os.makedirs(os.path.dirname(emb_memory_loc), exist_ok=True)
    # 文件创建（仅限 rank 0）
    if rank == 0:
        emd_memmap = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
        del emd_memmap  # 完成创建后释放资源
    # 同步所有进程
    dist.barrier()
    # 所有进程以 'r+' 模式打开文件
    emd_memmap = np.memmap(emb_memory_loc, dtype='float32', mode='r+', shape=(dataset_size, emb_size))
    def collate_fn(examples):
        input_ids = torch.tensor([ex['input_ids'] for ex in examples])
        attention_mask = torch.tensor([ex['attention_mask'] for ex in examples])
        index = [ex['idx'] for ex in examples]
        return input_ids,attention_mask,index

    # 加载Dataloader
    sampler=DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=192,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=8,
    )
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    ddp_model.eval()
    print(f"Get encoding on rank {rank}...")
    with torch.no_grad():
        for input_ids,attention_mask,index_batch in tqdm(dataloader):
            # 获取隐藏层状态
            input_ids = input_ids.to(device_id)
            attention_mask = attention_mask.to(device_id)
            outputs = ddp_model(input_ids, attention_mask=attention_mask,output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state
            # 找到每个样本的最后一个有效token的位置
            last_token_indices = (attention_mask.sum(dim=1) - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, last_hidden_state.size(-1))
            last_token_state = last_hidden_state.gather(1, last_token_indices).squeeze(1)
            # 对 last_token_state 进行处理，例如归一化
            normalized_last_layer_states = normalize(last_token_state, dim=1)
            # 存储embedding
            normalized_last_layer_states = normalized_last_layer_states.to(dtype=torch.float32).cpu().numpy()
            for i in range(len(index_batch)):
                emd_memmap[index_batch[i]]+=normalized_last_layer_states[i]
    if rank == 0:
        for i in tqdm(range(dataset_size)):
            emd_memmap[i]=emd_memmap[i]/index_num[i]
    # 清除并行集群
    dist.destroy_process_group()

if __name__  == '__main__':
    fire.Fire(main)

