import yaml
import numpy as np
import pandas as pd
import torch
import pickle
import random
import math
import time
import os
import pprint
import argparse
import sys
import submitit
from tqdm import tqdm
sys.path.append('/home/guochuanzhe/data-process/SemDeDup')
# 假设 constants 模块已经存在
from constants import DIST_METRIC_INDEX

# SemDeDup类的定义和其他函数（已经在之前的代码中提供）

def init_memmap_embs(
    embs_memory_loc: str, dataset_size: int, emb_size: int = 512, dtype: str = "float32"
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array to read embeddings of examples.

    Args:
        embs_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emb_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    embs = np.memmap(
        embs_memory_loc, dtype=dtype, mode="r", shape=(dataset_size, emb_size)
    )
    return embs


class SemDeDup():
    """
    - Each SLURMJob will run SemDeDup on number of clusters and save dataframe with which examples to keep from each cluster.
    - Parallelize job_start_cluster across jobs so that preemption in the middle of an epoch isn't a problem and because we want to
    keep the shard structure anyway.
    - Process more than one cluster per job=> run multiple taks inside each jobs.
    - Preempted jobs get resubmitted. Already precessed clusters get skipped internally.
    """

    def __init__(self, params):
        model_name=params['model_name']
        dataset_name=params['dataset_name']
        split=params['split']
        type=params['type']
        self.params = params
        self.emb_memory_loc=f"/home/guochuanzhe/data-process/SemDeDup/memory/embedding/{dataset_name}/{type}/{model_name}/emb_memory_loc.npy"
        self.dataset_size=params["dataset_size"]
        self.emb_size=params["emb_size"]
        self.sorted_clusters_path = f"/home/guochuanzhe/data-process/SemDeDup/memory/sorted_clusters_file/{dataset_name}/{type}/{model_name}"
        self.semdedup_save_folder=f"/home/guochuanzhe/data-process/SemDeDup/memory/semdedup/{dataset_name}/{type}/{model_name}"
        self.device="cuda" if torch.cuda.is_available() else "cpu"  
        self.eps_list=params['eps_list']
        self.which_to_keep=params['which_to_keep']

        random.seed(params['seed'])

    def _contains_duplicates(self, arr):
        return len(np.unique(arr)) != len(arr)

    def semdedup(self, cluster, cluster_reps, device):
        st = time.time()
        ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
        cluster_reps.to(device)
        pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
        del cluster_reps
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

        ## -- get paths to cluster i images
        image_urls = cluster[:, 0]

        ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
        assert not self._contains_duplicates(image_urls)

        ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

        ## -- if the max sim between one example and any other example is > 1-eps, remove this example
        M = torch.max(triu_sim_mat, dim=0)[0].cpu()
        print(f"Step time: {time.time()-st}(s)")

        return M

    def _process_shard(self, start_cluster: int, end_cluster: int):
        # print("SemDeDup params: ", self.params)
        st = time.time()

        embs = init_memmap_embs(
            self.emb_memory_loc, self.dataset_size, self.emb_size
        )

        step_time = []
        count=0
        for cluster_id in tqdm(range(start_cluster, end_cluster)):
            step_st = time.time()

            df_file_loc = os.path.join(
                self.semdedup_save_folder, f"cluster_{cluster_id}.pkl"
            )

            if os.path.exists(df_file_loc):  # and os.path.exists(dict_file_loc):
                print(f"{df_file_loc} exists, moving on")
                continue

            ## -- load cluster i representations
            cluster_i = np.load(
                os.path.join(
                    self.sorted_clusters_path, f"cluster_{cluster_id}.npy"
                )
            )
            # print(cluster_i.shape)
            # continue
            # 1) store cluster size
            cluster_size = cluster_i.shape[0]
            print("cluster_size: ", cluster_size)

            if cluster_size == 1:
                points_to_remove_df = pd.DataFrame()
                points_to_remove_df["indices"] = [0]
                for eps in self.eps_list:
                    ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                    points_to_remove_df[f"eps={eps}"] = [False]
                if self.semdedup_save_folder != "":
                    ## --save df
                    with open(df_file_loc, "wb") as file:
                        pickle.dump(points_to_remove_df, file)
                print("DONE cluster_id ", cluster_id)
                continue

            ## -- By default, we keep hard examples from groups
            clutser_items_indices = list(range(cluster_size))
            ## -- OR: shuffle cluster to keep random example from each group
            if self.which_to_keep.lower() == "random":
                random.shuffle(clutser_items_indices)
                cluster_i = cluster_i[clutser_items_indices]
            ## -- OR: reverse cluster to keep easy examples
            if self.which_to_keep.lower() == "easy":
                clutser_items_indices = clutser_items_indices[::-1]
                cluster_i = cluster_i[clutser_items_indices]

            ## -- indices for cluster items in the dataset
            cluster_ids = cluster_i[:, 1].astype("int32")
            cluster_reps = embs[cluster_ids]
            cluster_reps = torch.tensor(cluster_reps)

            M = self.semdedup(cluster_i, cluster_reps, self.device)

            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = clutser_items_indices

            for eps in self.eps_list:
                ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                eps_points_to_remove = M > 1 - eps
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

            if self.semdedup_save_folder != "":
                ## --save df
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)

            step_time.append(time.time() - step_st)
            print("DONE cluster: ", cluster_id)

        print(
            f"DONE in {((time.time()-st)/60):.2f} minutes, Average Step time {(sum(step_time)/len(step_time)):.2f}(s)"
        )
        # print(count)
        return
        

def main():
    # 读取配置文件
    params_file = "/home/guochuanzhe/data-process/SemDeDup/semdedup_configs.yaml"  # 配置文件路径
    with open(params_file, "r") as file:
        params=yaml.load(file,Loader=yaml.FullLoader)
    # SemDeDup
    semdedup = SemDeDup(params)
    semdedup._process_shard(0, params["num_clusters"])

if __name__ == "__main__":
    main()





