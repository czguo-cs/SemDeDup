
import numpy as np
import json
import yaml
import time
from multiprocessing import Pool

def process_eps(eps, dataset_name, split, npy_folder_path, json_file_path, json_output_folder_path, retained_data_ratios):
    print(f"Start processing for eps {eps}")
    start_time = time.time()
    # 计算数据比例后缀
    ratio_suffix = f"{(retained_data_ratios[str(eps)]/100):.2f}"

    # 加载 NumPy 数据
    npy_file_path = f"{npy_folder_path}/keep_data_{eps}.npy"
    example_data = set(np.load(npy_file_path))

    # 加载整个 JSON 文件到内存
    with open(json_file_path, 'r') as file:
        json_data = file.readlines()

    # 过滤 JSON 数据
    json_output_path = f"{json_output_folder_path}/starcoder-{dataset_name}-{split}-2048-dedup_{eps}_{ratio_suffix}.jsonl"
    with open(json_output_path, 'w') as file:
        for line_number, line in enumerate(json_data):
            if line_number in example_data:
                json_obj = json.loads(line)
                json.dump(json_obj, file)
                file.write('\n')

    end_time = time.time()
    print(f"DONE processing for eps {eps}. Time taken: {end_time - start_time:.2f} seconds.")

def generate_filtered_json(dataset_name, split, npy_folder_path, json_file_path, json_output_folder_path, save_eps_list):
    # 读取保留数据比例
    with open(npy_folder_path+"/retained_data_ratios.json", "r") as file:
        retained_data_ratios = json.load(file)
    # 使用多进程进行处理
    with Pool(len(save_eps_list)) as pool:
        pool.starmap(process_eps, [(eps, dataset_name, split, npy_folder_path, json_file_path, json_output_folder_path, retained_data_ratios) for eps in save_eps_list])

# 这里添加代码以调用 generate_filtered_json 函数
if __name__=='__main__':
    confg_file = "../../semdedup_configs.yaml"
    ## -- Load kmeans clustering parameters from configs file
    with open(confg_file, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    model_name=params['model_name']
    dataset_name=params['dataset_name']
    split=params['split']
    type=params['type']

    # 原始数据路径
    json_file_path = "your/path/origin_data.jsonl"

    # json_file_path=f"/home/guochuanzhe/data-process/data_reorganization/data/{dataset_name}/starcoder-{dataset_name}-{split}-2048/llama-350M/starcoder-{dataset_name}-{split}-2048.jsonl"
    # json_file_path = f"/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-{dataset_name}-{split}.jsonl"
    json_output_folder_path=f"../../data/{dataset_name}/pruned-starcoder-{dataset_name}-{split}/{type}/{model_name}"
    output_npy_path = f"../../memory/output_path/{dataset_name}/{type}/{model_name}"
    generate_filtered_json(
        dataset_name=dataset_name,
        split=split,
        npy_folder_path=output_npy_path,
        json_file_path=json_file_path,
        json_output_folder_path=json_output_folder_path,
        save_eps_list=params['save_eps_list'], 
    )
