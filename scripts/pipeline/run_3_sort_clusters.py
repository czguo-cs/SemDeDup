import yaml
import random
import numpy as np
import sys
sys.path.append('/home/guochuanzhe/data-process/SemDeDup/clustering')
from sort_clusters import assign_and_sort_clusters



confg_file = "/home/guochuanzhe/data-process/SemDeDup/semdedup_configs.yaml"
## -- Load kmeans clustering parameters from configs file
with open(confg_file, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

## -- Fix the seed
random.seed(params['seed'])

# 获取路径变量
model_name=params['model_name']
dataset_name=params['dataset_name']
split=params['split']

sorted_clusters_path = f"/home/guochuanzhe/data-process/SemDeDup/memory/sorted_clusters_file/{dataset_name}/{model_name}"
kmeans_save_folder=f"/home/guochuanzhe/data-process/SemDeDup/memory/kmeans_save_folder/{dataset_name}/{model_name}"
emb_memory_loc=f"/home/guochuanzhe/data-process/SemDeDup/memory/embedding/{dataset_name}/{model_name}/emb_memory_loc.npy"

# 获取embedding_memory
dataset_size = params['dataset_size'] 
emb_size = params['emb_size'] 
emb_memory = np.memmap(emb_memory_loc, dtype='float32', mode='r', shape=(dataset_size, emb_size))
paths_memory = np.array(range(dataset_size))



assign_and_sort_clusters(
    data=emb_memory,
    paths_list=paths_memory,
    sim_metric=params["sim_metric"],
    keep_hard=params["keep_hard"],
    kmeans_with_cos_dist=params["kmeans_with_cos_dist"],
    save_folder=kmeans_save_folder,
    sorted_clusters_file_loc=sorted_clusters_path,
    cluster_ids=range(0, params["num_clusters"]),
) 

