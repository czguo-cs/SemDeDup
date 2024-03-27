import numpy as np
import multiprocessing
from torch.utils.data import DataLoader 
import sys
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




# # # 训练函数
# # def get_nl_embeddings(model, dataloader, emd_memmap, paths_memmap):
# #     """
# #     function to compute and store representations for the data from pretrained model. It is preferable to parallelize this function on mulitiple devices (GPUs). Each device will process part of the data.
# #     model: pretrained model
# #     dataloader: should return   1) data_batch: batch of data examples
# #                                 2) paths_batch: path to location where the example is stored (unique identifier). For example, this could be "n04235860_14959.JPEG" for imagenet.
# #                                 3) batch_indices: global index for each example (between 0 and of size <dataset_size>-1).
# #     emd_memmap: numpy memmap to store embeddings of size <dataset_size>.
# #     paths_memmap: numpy memmap to store paths of size <dataset_size>.

# #     """

# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     model = model.to(device)
# #     model.eval()
# #     print("Get encoding...")
# #     with torch.no_grad():
# #         for data_batch,index_batch in tqdm(dataloader):
# #             data_batch = data_batch.to(device)
# #             # print(data_batch)
# #             # 获取隐藏层状态
# #             input_ids = data_batch['input_ids'].to(device)
# #             attention_mask = data_batch['attention_mask'].to(device)
# #             outputs = model(input_ids, attention_mask=attention_mask,output_hidden_states=True)
# #             last_hidden_state = outputs.last_hidden_state
# #             last_token_state = last_hidden_state[:, -1, :]
# #             # 您可以在此处对 last_layer_states 进行处理，例如归一化
# #             normalized_last_layer_states = normalize(last_token_state, dim=1)
# #             # # 存储embedding
# #             normalized_last_layer_states = normalized_last_layer_states.to(dtype=torch.float32).cpu().numpy()
# #             for i in range(len(index_batch)):
# #                 emd_memmap[index_batch[i]]=normalized_last_layer_states[i]
# #                 paths_memmap[index_batch[i]]=index_batch[i]



# # 加载数据集
# class JsonlDataset(Dataset):
#     def __init__(self, file_path):
#         self.data = []
#         with open(file_path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 self.data.append(json.loads(line))
#         # self.index=range(len(self.data))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # 这里假设 JSONL 文件中每行是一个包含文本的 JSON 对象
#         text = self.data[idx]['content']  # 根据实际数据结构调整
#         return text,idx


# # 定义DataLoader
# path_data = "/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-sql-dev.jsonl"
# dataset=JsonlDataset(path_data)


# model_path="/home/guochuanzhe/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"

# # 加载OPT-125模型及其分词器
# tokenizer = AutoTokenizer.from_pretrained(
#     model_path,
#     padding_side="right",
#     use_fast=True, 
# )
# model = AutoModel.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2"
# )

# # 定义collate_fn
# def collate_fn(examples):
#     texts = [ex[0] for ex in examples]
#     index_batch = [ex[1] for ex in examples]
#     data_batch = tokenizer(texts, return_tensors="pt",padding=True,truncation=True, max_length=2048)
#     return data_batch,index_batch
    

# # # 加载Dataloader
# # sampler=DistributedSampler(dataset)

# # dataloader = DataLoader(
# #     dataset=dataset,
# #     batch_size=64,
# #     shuffle=False,
# #     sampler=sampler,
# #     collate_fn=collate_fn,
# #     num_workers=36,
# # )

# path_str_type = int
# emb_memory_loc = "/home/guochuanzhe/data-process/SemDeDup/memory/embedding/emb_memory_loc.npy"
# paths_memory_loc = "/home/guochuanzhe/data-process/SemDeDup/memory/embedding/paths_memory.npy"  
# dataset_size = len(dataset)                                                          # starcoder
# emb_size = 768
# emd_memmap = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
# # paths_memmap = np.memmap(paths_memory_loc, dtype=path_str_type, mode='w+', shape=(dataset_size,))


# get_nl_embeddings(model, dataloader, emd_memmap, paths_memmap)



# 加载数据集
class JsonlDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                if 'content' in json_data:  # 假设JSON数据中有一个名为'text'的字段
                    text = json_data['content'][:100]  # 截取前100个字符
                    json_data['content'] = text  # 更新文本字段
                self.data.append(json_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 这里假设 JSONL 文件中每行是一个包含文本的 JSON 对象
        text = self.data[idx]['content']  # 根据实际数据结构调整
        return text,idx



def main():
    # dist.init_process_group("nccl")
    # rank = dist.get_rank()
    
    # print(f"Start running basic DDP example on rank {rank}.")
    emb_memory_loc = "/home/guochuanzhe/data-process/SemDeDup/scripts/test/emb_memory_loc.npy"
    # emb_memory_loc = "/home/guochuanzhe/data-process/SemDeDup/memory/embedding/python/llama-350M/emb_memory_loc.npy"
    path_data = "/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-sql-dev.jsonl"
    # dataset=JsonlDataset(path_data)
    dataset_size = 10                                                        # starcoder
    emb_size = 768
    # if dist.get_rank()==0:
    emd_memmap = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
    # else:
    #     emd_memmap = np.memmap(emb_memory_loc, dtype='float32', mode='r+', shape=(dataset_size, emb_size))
    for i in range(10):
        emd_memmap[i]+=np.ones(768)
        print(emd_memmap[i])
    # model_path="/home/guochuanzhe/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"
    # model_path="/home/guochuanzhe/model/llama/pre_train/llama-350M/llama-350M-MONO/325/iter0005812"
    # 加载OPT-125模型及其分词器
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_path,
    #     padding_side="right",
    #     use_fast=True, 
    # )
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens(special_tokens_dict=dict(pad_token="</s>"))
    # model = AutoModel.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2"
    # )

    # # 定义collate_fn
    # def collate_fn(examples):
    #     texts = [ex[0] for ex in examples]
    #     index_batch = [ex[1] for ex in examples]
    #     data_batch = tokenizer(texts, return_tensors="pt",padding=True,truncation=True, max_length=2048)
    #     return data_batch,index_batch
        

    # # 加载Dataloader
    # # sampler=DistributedSampler(dataset)

    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=2,
    #     shuffle=False,
    #     # sampler=sampler,
    #     collate_fn=collate_fn,
    #     num_workers=4,
    # )
    # # get_nl_embeddings(model, dataloader, emd_memmap, paths_memmap)
    # # create model and move it to GPU with id rank
    # # "cuda" = rank % torch.cuda.device_count()
    # # 加载OPT-125模型及其分词器
    # model = model.to("cuda")
    # # model = DDP(model, "cuda"s=["cuda"])
    # model.eval()
    # print(f"Get encoding on rank {rank}...")
    count=0
    # with torch.no_grad():
    #     for data_batch, index_batch in tqdm(dataloader):
    #         # 获取隐藏层状态
    #         input_ids = data_batch['input_ids'].to("cuda")
    #         attention_mask = data_batch['attention_mask'].to("cuda")

    #         outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #         last_hidden_state = outputs.last_hidden_state

    #         # 找到每个样本的最后一个有效token的位置
    #         last_token_indices = (attention_mask.sum(dim=1) - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, last_hidden_state.size(-1))
    #         last_token_state = last_hidden_state.gather(1, last_token_indices).squeeze(1)

    #         # 对 last_token_state 进行处理，例如归一化
    #         normalized_last_layer_states = normalize(last_token_state, dim=1)

    #         # 存储embedding
    #         normalized_last_layer_states = normalized_last_layer_states.to(dtype=torch.float32).cpu().numpy()
    #         count += 1
    #         if count==1:
    #             # print(input_ids)
    #             # print(last_token_state)
    #             # print(last_hidden_state[:,-3,:])
    #             print(emd_memmap[count-1])
    #             break

            # for i in range(len(index_batch)):
            #     emd_memmap[index_batch[i]] = normalized_last_layer_states[i]
                # print(normalized_last_layer_states)
                # print(f"index_batch size{len(index_batch)}")
                # print(index_batch)
                # for i in range(len(index_batch)):
                #     emd_memmap[index_batch[i]]=normalized_last_layer_states[i]
                    # print(f"emd_memmap[index_batch[i]] size{len(emd_memmap[index_batch[i]])}")
                    # print(emd_memmap[index_batch[i]])
                    # paths_memmap[index_batch[i]]=index_batch[i]

    # outputs = model(torch.randn(20, 10))
    # labels = torch.randn(20, 5).to("cuda")
    # loss_fn(outputs, labels).backward()
    # optimizer.step()
    # dist.destroy_process_group()

if __name__  == '__main__':
    # paths_memmap = np.memmap(paths_memory_loc, dtype=path_str_type, mode='w+', shape=(dataset_size,))
    main()

main()