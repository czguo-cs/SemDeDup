
import os
import numpy as np
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import time
import json
from constants import IMAGE_NAME_INDEX

def process_cluster(cluster_id, sorted_clusters_path, semdedup_save_folder, eps, retreive_kept_samples):
    cluster_i = np.load(
        os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
    )
    with open(f"{semdedup_save_folder}/cluster_{cluster_id}.pkl", "rb") as file:
        semdedup_pruning_tables = pickle.load(file)

    images_to_keep_or_remove = semdedup_pruning_tables[f"eps={eps}"][
        semdedup_pruning_tables[f"eps={eps}"] == (not retreive_kept_samples)
    ].index.to_numpy()
    if "indices" in semdedup_pruning_tables.columns:
        cluster_i = cluster_i[semdedup_pruning_tables["indices"]]

    dedup_cluster = cluster_i[images_to_keep_or_remove]
    return dedup_cluster[:, IMAGE_NAME_INDEX]

def extract_pruned_data(
    dataset_size:int,
    sorted_clusters_path:str,
    semdedup_save_folder:str,
    eps_list:list,
    num_clusters:int,
    output_npy_path:str,
    retreive_kept_samples:bool = True,
    num_processes:int = 4
):
    start_time = time.time()
    retained_data_ratios = {}  # 用于保存每个eps的保留数据比例
    for eps in eps_list:
        print(f"now start to process eps {eps}")

        with Pool(num_processes) as p:
            results = list(tqdm(p.starmap(process_cluster, [(cluster_id, sorted_clusters_path, semdedup_save_folder, eps, retreive_kept_samples) for cluster_id in range(num_clusters)]), total=num_clusters))
        
        example_data = set(np.concatenate(results))
        np.save(output_npy_path+f"/keep_data_{eps}.npy", list(example_data))

        retained_ratio = len(example_data) * 100 / dataset_size
        retained_data_ratios[eps] = retained_ratio
        print(f"DONE saving {len(example_data)} data entries, Remaining Data {retained_ratio}%")

    # 保存保留数据比例
    with open(output_npy_path+"/retained_data_ratios.json", "w") as file:
        json.dump(retained_data_ratios, file)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    return

