# # file: generate_filtered_json.py

# import numpy as np
# import json
# import yaml
# import time
# from tqdm import tqdm

# def generate_filtered_json(
#     npy_folder_path:str,
#     json_file_path:str,
#     json_output_folder_path:str,
#     save_eps_list:list,
# ):
#     for eps in save_eps_list:
#         print(f"Start save dedup JSON entries for eps {eps}")
#         start_time = time.time()  # 开始计时
#         npy_file_path = f"{npy_folder_path}/keep_data_{eps}.npy"
#         json_output_path = f"{json_output_folder_path}/dedup_data_{eps}.jsonl"

#         example_data = set(np.load(npy_file_path))

#         # -- 过滤 JSON 数据
#         filtered_json_data = []
#         with open(json_file_path, 'r') as file:
#             for line_number, line in enumerate(file):
#                 if line_number in example_data:
#                     json_obj = json.loads(line)
#                     filtered_json_data.append(json_obj)

#         # -- 保存过滤后的 JSON 数据
#         with open(json_output_path, 'w') as file:
#             for item in filtered_json_data:
#                 json.dump(item, file)
#                 file.write('\n')
#         end_time = time.time()  # 结束计时
#         print(f"DONE saving {len(filtered_json_data)} filtered JSON entries for eps {eps}. Time taken: {end_time - start_time:.2f} seconds.")

#     return

# # 这里添加代码以调用 generate_filtered_json 函数
# if __name__=='__main__':
#     confg_file = "/home/guochuanzhe/data-process/SemDeDup/semdedup_configs.yaml"
#     ## -- Load kmeans clustering parameters from configs file
#     with open(confg_file, 'r') as y_file:
#         params = yaml.load(y_file, Loader=yaml.FullLoader)
#     model_name=params['model_name']
#     dataset_name=params['dataset_name']
#     split=params['split']
#     json_file_path = f"/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-{dataset_name}-{split}.jsonl"
#     json_output_folder_path=f"/home/guochuanzhe/data-process/SemDeDup/data/{dataset_name}/pruned-starcoder-{dataset_name}-{split}/{model_name}"
#     print(params['save_eps_list'])
#     print(json_output_folder_path)
#     generate_filtered_json(
#         npy_folder_path=params['output_npy_path'],
#         json_file_path=json_file_path,
#         json_output_folder_path=json_output_folder_path,
#         save_eps_list=params['save_eps_list'], 
#     )
    
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
    confg_file = "/home/guochuanzhe/data-process/SemDeDup/semdedup_configs.yaml"
    ## -- Load kmeans clustering parameters from configs file
    with open(confg_file, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    model_name=params['model_name']
    dataset_name=params['dataset_name']
    split=params['split']
    type=params['type']
    json_file_path=f"/home/guochuanzhe/data-process/data_reorganization/data/{dataset_name}/starcoder-{dataset_name}-{split}-2048/llama-350M/starcoder-{dataset_name}-{split}-2048.jsonl"
    # json_file_path = f"/home/guochuanzhe/Megatron-LM-gjn/data/starcoder/chatml_data/starcoder-{dataset_name}-{split}.jsonl"
    json_output_folder_path=f"/home/guochuanzhe/data-process/SemDeDup/data/{dataset_name}/pruned-starcoder-{dataset_name}-{split}/{type}/{model_name}"
    output_npy_path = f"/home/guochuanzhe/data-process/SemDeDup/memory/output_path/{dataset_name}/{type}/{model_name}"
    generate_filtered_json(
        dataset_name=dataset_name,
        split=split,
        npy_folder_path=output_npy_path,
        json_file_path=json_file_path,
        json_output_folder_path=json_output_folder_path,
        save_eps_list=params['save_eps_list'], 
    )
