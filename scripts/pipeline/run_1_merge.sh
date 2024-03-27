#!/bin/bash
#SBATCH -J compute_embedding                                       # 作业名为 test
#SBATCH -o /home/guochuanzhe/data-process/SemDeDup/logs/merge_test-%j.log                      # stdout 重定向到 test.out
#SBATCH -e /home/guochuanzhe/data-process/SemDeDup/logs/merge_test-%j.err                     # stderr 重定向到 test.err
#SBATCH -p gpu4                                                         # 作业提交的分区为 compute
#SBATCH -N 1                                                          # 作业申请 1 个节点
#SBATCH -w g4006      
#SBATCH -x g4007,g4008                                                  # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:8
#SBATCH --mem 970GB
#SBATCH -c 128

. "/home/guochuanzhe/anaconda/etc/profile.d/conda.sh"
conda activate semdedup


opt_125M_model_path="/home/guochuanzhe/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6" 
llama_350M_model_path="/home/guochuanzhe/model/llama/pre_train/llama-350M/llama-350M-MONO/325/iter0005812"
type="merge"



# python run_1_tokenizer_merge.py \
#     --model_path $llama_350M_model_path \
#     --model_name llama-350M \
#     --dataset_name python \
#     --split train \
#     --type $type \
#     --emb_size 1024


export OMP_NUM_THREADS=10
torchrun --nproc_per_node 8 --node_rank=$SLURM_NODEID --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=0 --rdzv_endpoint=localhost:45426 run_1_get_embedding_merge.py \
    --model_path $llama_350M_model_path \
    --model_name llama-350M \
    --dataset_name python \
    --split train \
    --type $type \
    --emb_size 1024

