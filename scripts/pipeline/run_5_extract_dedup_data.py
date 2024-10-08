import sys
import yaml
import argparse
sys.path.append("../..")
from extract_dedup_data import extract_pruned_data


confg_file = "../../semdedup_configs.yaml"
## -- Load kmeans clustering parameters from configs file
with open(confg_file, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

model_name=params['model_name']
dataset_name=params['dataset_name']
split=params['split']
type=params['type']

sorted_clusters_path = f"../../memory/sorted_clusters_file/{dataset_name}/{type}/{model_name}"
semdedup_save_folder = f"../../memory/semdedup/{dataset_name}/{type}/{model_name}"
output_npy_path = f"../../memory/output_path/{dataset_name}/{type}/{model_name}"

extract_pruned_data(
    dataset_size = params['dataset_size'],
    sorted_clusters_path=sorted_clusters_path, 
    semdedup_save_folder=semdedup_save_folder, 
    eps_list=params['eps_list'], 
    num_clusters=params['num_clusters'],
    output_npy_path=output_npy_path, 
    retreive_kept_samples=True,
    num_processes=8
)