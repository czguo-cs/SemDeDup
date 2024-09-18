# SemDeDup

## 环境配置
参考https://github.com/facebookresearch/SemDeDup

## 执行
执行文件为./scripts/pipeline/run_2_3_4_5_6.sh

步骤一为求embedding，已有embedding可省略此步骤，将embedding以.npy格式存储在./memory/embedding目录下

## 需要修改的参数
1. ./scripts/pipeline/run_2_3_4_5_6.sh
```
# 以下参数需要修改，主要用于目录及文件命名命名，可自主修改
# 与../../semdedup_configs.yaml保持一致
model_name=llama-350M
dataset_name=code_sft
split=train
type=sft
```

2. ../../semdedup_configs.yaml
```
# 用于路径或文件命名的参数
model_name: "llama-350M"
dataset_name: "code_sft"
split: "train"
type: "sft"
# -- number of clusters
num_clusters: 100
# 待处理数据集的大小
dataset_size: 486439
# -- embeddings size
emb_size: 1024
# 保留规则，不需要修改
which_to_keep: "hard"
# -- seed
seed: 1234
# -- largest cluster size the memory is large enough to process. If the cluster size is larger than it, we will devide the cluster into small clusters and process each one separately.
largest_cluster_size_to_process: 10000000
sim_metric: 'cosine' # choose form ['cosine', 'l2']
keep_hard: True # True for hard examples
kmeans_with_cos_dist: True # True for using cosine similarity for kmeans clustering

# k-means聚类迭代次数，>=收敛迭代数
niter: 150
# 阈值列表
eps_list: [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07]
# 选择合适的阈值保存结果
save_eps_list: [0.01]
```


3. ../../scripts/pipeline/run_6_generate_dedup_json.py
```
# 原始数据路径
json_file_path = "your/path/origin_data.jsonl"
```

## 注意事项
1. 目前的路径是按照我原本的规则命名的，建议修改以方便使用