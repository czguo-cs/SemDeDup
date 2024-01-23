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



# # 一次性加载
# # JsonlDataset类，继承Dataset，从jsonl文件获取数据
class JsonlDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 这里假设 JSONL 文件中每行是一个包含文本的 JSON 对象
        text = self.data[idx]['content']  # 根据实际数据结构调整
        return text,idx

# # 惰性加载(不好用，慢的离谱)
# class JsonlDataset(Dataset):
#     def __init__(self, file_path):
#         self.file_path = file_path

#     def __len__(self):
#         return 8519717

#     def __getitem__(self, idx):
#         with open(self.file_path, 'r') as file:
#             for i, line in enumerate(file):
#                 if i == idx:
#                     data = json.loads(line)
#                     break
#         text = data['content']
#         return text, idx



def main(
    model_path:str = "/home/guochuanzhe/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    model_name:str = "opt-125M",
    dataset_name:str = "python",
    split:str = "train",
    emb_size:int = 768,  
):
    # 并行初始化
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    # 基础元素初始化
    print(f"Start running model {model_name} for dataset {dataset_name}-{split} on rank {rank}.")
    emb_memory_loc = f"/home/guochuanzhe/data-process/SemDeDup/memory/embedding/{dataset_name}/merge/{model_name}/emb_memory_loc.npy"
    path_data = f"/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-{dataset_name}-{split}.jsonl"
    dataset=JsonlDataset(path_data)
    dataset_size = len(dataset)                                                         # starcoder
    # 创建父目录（如果不存在）
    os.makedirs(os.path.dirname(emb_memory_loc), exist_ok=True)
    # 文件创建（仅限 rank 0）
    if dist.get_rank() == 0:
        emd_memmap = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
        del emd_memmap  # 完成创建后释放资源
    # 同步所有进程
    dist.barrier()
    # 所有进程以 'r+' 模式打开文件
    emd_memmap = np.memmap(emb_memory_loc, dtype='float32', mode='r+', shape=(dataset_size, emb_size))

    # tokenizer and model 
    print(f"start load model and tokenizer on rank {rank}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="</s>"))
    # 与原来相比，opt-125M由f16改为bf16
    if model_name == "opt-125M":
        torch_dtype=torch.float16
    else:
        torch_dtype=torch.bfloat16
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2"
    )
    print(f"model and tokenizer loaded successfully on rank {rank}")

    # 定义collate_fn
    # def collate_fn(examples):
    #     texts = [ex[0] for ex in examples]
    #     index_batch = [ex[1] for ex in examples]
    #     data_batch = tokenizer(texts, return_tensors="pt",padding=True,truncation=True, max_length=2048)
    #     return data_batch,index_batch
    

    def collate_fn(examples, max_length=2048):
        texts = [ex[0] for ex in examples]
        index_batch = [ex[1] for ex in examples]
        split_texts = []
        split_index_batch = []
        for text, index in zip(texts, index_batch):
            # 分割文本
            for i in range(0, len(text), max_length):
                split_texts.append(text[i:i + max_length])
                split_index_batch.append(index)
        data_batch = tokenizer(split_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        return data_batch, split_index_batch    

    # 加载Dataloader
    sampler=DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=1,
    )
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    ddp_model.eval()
    print(f"Get encoding on rank {rank}...")
    with torch.no_grad():
        for data_batch,index_batch in tqdm(dataloader):
            # 获取隐藏层状态
            input_ids = data_batch['input_ids'].to(device_id)
            attention_mask = data_batch['attention_mask'].to(device_id)
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
    
    # 清除并行集群
    dist.destroy_process_group()

if __name__  == '__main__':
    fire.Fire(main)

