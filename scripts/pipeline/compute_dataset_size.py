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



# 加载数据集
class JsonlDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.data.append(json.loads(line))
        # self.index=range(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 这里假设 JSONL 文件中每行是一个包含文本的 JSON 对象
        text = self.data[idx]['content']  # 根据实际数据结构调整
        return text,idx


# 实例化数据集并求长度
text = []
path_data = "/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-sql-train.jsonl"
dataset1=JsonlDataset(path_data)
text.append(f"the size of sql-train dataset is {len(dataset1)} ")
path_data = "/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-python-train.jsonl"
dataset2=JsonlDataset(path_data)
text.append(f"the size of python-train dataset is {len(dataset2)} ")
# 打开文件，如果文件不存在则创建
with open("example.txt", "w") as file:
    for textline in  text:
        file.write(textline + "\n")
