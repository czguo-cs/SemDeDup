import yaml
import random
import numpy as np
import sys
sys.path.append('/home/guochuanzhe/data-process/SemDeDup/clustering')
from clustering import compute_centroids


confg_file = "/home/guochuanzhe/data-process/SemDeDup/semdedup_configs.yaml"
## -- Load kmeans clustering parameters from configs file
with open(confg_file, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

## -- Fix the seed
SEED = params['seed']
random.seed(SEED)

# 获取路径变量
model_name=params['model_name']
dataset_name=params['dataset_name']
split=params['split']
type=params['type']
kmeans_save_folder=f"/home/guochuanzhe/data-process/SemDeDup/memory/kmeans_save_folder/{dataset_name}/{type}/{model_name}"
emb_memory_loc=f"/home/guochuanzhe/data-process/SemDeDup/memory/embedding/{dataset_name}/{type}/{model_name}/emb_memory_loc.npy"


# 获取embedding_memory
dataset_size = params['dataset_size'] 
emb_size = params['emb_size'] 
emb_memory = np.memmap(emb_memory_loc, dtype='float32', mode='r', shape=(dataset_size, emb_size))

compute_centroids(
    data=emb_memory,
    ncentroids=params['num_clusters'],
    niter=params['niter'],
    seed=params['seed'],
    Kmeans_with_cos_dist=params['kmeans_with_cos_dist'],
    save_folder=kmeans_save_folder,
    verbose=True,
)