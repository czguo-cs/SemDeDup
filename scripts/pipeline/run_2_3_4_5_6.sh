#!/bin/bash
#SBATCH -J semdedup                                       # 作业名为 test
#SBATCH -o .../logs/semdedup-%j.log                     # stdout 重定向到 test.out
#SBATCH -e ../../logs/semdedup-%j.err                   # stderr 重定向到 test.err
#SBATCH -p hit                                                        # 作业提交的分区为 compute
#SBATCH -N 1                                                          # 作业申请 1 个节点
# SBATCH -w g4001      
# SBATCH -x g4007,g4008                                                  # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
# SBATCH --gres=gpu:8
#SBATCH --mem 100GB
#SBATCH -c 32
. "/home/guochuanzhe/anaconda/etc/profile.d/conda.sh"
conda activate semdedup


# 以下参数需要修改，主要用于目录及文件命名命名，可自主修改
# 与../../semdedup_configs.yaml保持一致
model_name=llama-350M
dataset_name=code_sft
split=train
type=sft


# run_2 compute_centroids
mkdir -p ../../memory/kmeans_save_folder/$dataset_name/$type/$model_name
rm -rf ../../memory/kmeans_save_folder/$dataset_name/$type/$model_name/*
python run_2_compute_centroids.py\

# # # # # # run_3 sort_clusters
mkdir -p ../../memory/sorted_clusters_file/$dataset_name/$type/$model_name
rm -rf ../../memory/sorted_clusters_file/$dataset_name/$type/$model_name/*
python run_3_sort_clusters.py\

# # # # run_4  SemDeDup
mkdir -p ../../memory/semdedup/$dataset_name/$type/$model_name
rm -rf ../../memory/semdedup/$dataset_name/$type/$model_name/*
python run_4_SemDeDup.py

# # # run_5 extract_dedup_data
mkdir -p ../../memory/output_path/$dataset_name/$type/$model_name
rm -rf ../../memory/output_path/$dataset_name/$type/$model_name/*
python run_5_extract_dedup_data.py

# run_6 generate_dedup_json
mkdir -p ../../data/$dataset_name/pruned-starcoder-$dataset_name-$split/$type/$model_name
rm -rf ../../data/$dataset_name/pruned-starcoder-$dataset_name-$split/$type/$model_name/*
python run_6_generate_dedup_json.py
