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
import pickle


# 加载数据集
class JsonlDataset(Dataset):
    def __init__(self, 
                 file_path,
                 tokenizer,
                 max_length=2048):
        self.input_ids = []
        self.attention_mask = []
        self.index =[]
        self.index_num=[]
        self.tokenizer=tokenizer
        count = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                data_line = json.loads(line)
                text = data_line['content'] if 'content' in data_line else data_line['text']
                data = self.tokenizer(text, truncation=False, return_tensors="pt", padding=False)
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
                self.index_num.append(input_ids_chunks.size()[0])
                for i in range(input_ids_chunks.size()[0]):
                    self.input_ids.append(input_ids_chunks[i])
                    self.attention_mask.append(attention_mask_chunks[i])
                    self.index.append(count)
                count+=1


    def __len__(self):
        return len(self.index_num)
    
    def get_index_num(self):
        return self.index_num
    
    def __getitem__(self, idx):
        # 这里假设 JSONL 文件中每行是一个包含文本的 JSON 对象
        return self.input_ids[idx],self.attention_mask[idx],self.index[idx]

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
    dataset_loc = f"/home/guochuanzhe/data-process/SemDeDup/memory/embedding/{dataset_name}/{type}/{model_name}/jsonl_dataset.pkl"
    emb_memory_loc = f"/home/guochuanzhe/data-process/SemDeDup/memory/embedding/{dataset_name}/{type}/{model_name}/emb_memory_loc.npy"
    path_data = f"/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-{dataset_name}-{split}.jsonl"
    # tokenizer and model 
    print(f"start load model on rank {rank}")
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
    print(f"model loaded successfully on rank {rank}")
    # 从文件加载对象
    with open(dataset_loc, 'rb') as file:
        dataset = pickle.load(file)
    print(f"dataset loaded successfully")
    dataset_size = len(dataset)   
    print(f"dataset_size:{dataset_size}")                                                 # starcoder
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
    def collate_fn(examples):
        input_ids = torch.stack([ex[0] for ex in examples])
        attention_mask = torch.stack([ex[1] for ex in examples])
        index = [ex[2] for ex in examples]
        return input_ids,attention_mask,index

    # 加载Dataloader
    sampler=DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2
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
        index_num =  dataset.get_index_num()  
        for i in tqdm(range(dataset_size)):
            emd_memmap[i] = emd_memmap[i] / index_num[i]
    # 清除并行集群
    dist.destroy_process_group()

if __name__  == '__main__':
    fire.Fire(main)

