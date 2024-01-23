import numpy as np
from torch.utils.data import DataLoader 
import sys
sys.path.append("/home/guochuanzhe/data-process/SemDeDup")
from compute_pretrained_embeddings import get_embeddings,get_nl_embeddings
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import Dataset
import json
import torch
from tqdm.auto import tqdm
from torch.nn.functional import normalize
# 加载数据集
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
        return text



# 定义DataLoader
path_data = "/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-sql-train.jsonl"
dataset=JsonlDataset(path_data)
model_path="/home/guochuanzhe/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"

# 加载OPT-125模型及其分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)


# 定义collate_fn
def collate_fn(examples):
    inputs = tokenizer(examples, return_tensors="pt",padding=True,truncation=True, max_length=2048)
    return inputs

# 加载Dataloader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

path_str_type = int
emb_memory_loc = "/home/guochuanzhe/data-process/SemDeDup/memory/embedding/emb_memory_loc.npy"
paths_memory_loc = "/home/guochuanzhe/data-process/SemDeDup/memory/embedding/paths_memory.npy"  
dataset_size = len(dataset)                                                           # starcoder
emb_size = 768
emd_memmap = np.memmap(emb_memory_loc, dtype='float32', mode='r',shape=(dataset_size,emb_size))
# paths_memmap = np.memmap(paths_memory_loc, dtype=path_str_type, mode='r', shape=(dataset_size,))


print(emd_memmap[0])
print(emd_memmap.shape)
