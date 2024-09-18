# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm.auto import tqdm
from torch.nn.functional import normalize


def get_embeddings(model, dataloader, emd_memmap, paths_memmap):
    """
    function to compute and store representations for the data from pretrained model. It is preferable to parallelize this function on mulitiple devices (GPUs). Each device will process part of the data.
    model: pretrained model
    dataloader: should return   1) data_batch: batch of data examples
                                2) paths_batch: path to location where the example is stored (unique identifier). For example, this could be "n04235860_14959.JPEG" for imagenet.
                                3) batch_indices: global index for each example (between 0 and of size <dataset_size>-1).
    emd_memmap: numpy memmap to store embeddings of size <dataset_size>.
    paths_memmap: numpy memmap to store paths of size <dataset_size>.
    """
    
    # -- Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- model
    model = model.to(device)
    model = model.eval()

    # -- Get and store 1)encodings 2)path to each example
    print("Get encoding...")
    with torch.no_grad():
        for data_batch, paths_batch, batch_indices in tqdm(dataloader):
            data_batch = data_batch.to(device)
            encodings = model(data_batch)
            emd_memmap[batch_indices] = normalize(encodings, dim=1)
            paths_memmap[batch_indices] = paths_batch
 
def get_nl_embeddings(model, dataloader, emd_memmap, paths_memmap):
    """
    function to compute and store representations for the data from pretrained model. It is preferable to parallelize this function on mulitiple devices (GPUs). Each device will process part of the data.
    model: pretrained model
    dataloader: should return   1) data_batch: batch of data examples
                                2) paths_batch: path to location where the example is stored (unique identifier). For example, this could be "n04235860_14959.JPEG" for imagenet.
                                3) batch_indices: global index for each example (between 0 and of size <dataset_size>-1).
    emd_memmap: numpy memmap to store embeddings of size <dataset_size>.
    paths_memmap: numpy memmap to store paths of size <dataset_size>.

    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    count =0 
    batch_size=1
    print("Get encoding...")
    with torch.no_grad():
        for data_batch in tqdm(dataloader):
            data_batch = data_batch.to(device)
            # print(data_batch)
            # 获取隐藏层状态
            input_ids = data_batch['input_ids'].to(device)
            attention_mask = data_batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state
            last_token_state = last_hidden_state[:, -1, :]
            # 您可以在此处对 last_layer_states 进行处理，例如归一化
            normalized_last_layer_states = normalize(last_token_state, dim=1)
            # # 存储embedding
            normalized_last_layer_states = normalized_last_layer_states.to(dtype=torch.float32).cpu().numpy()
            # # for i in range(count*batch_size,(count+1)*batch_size):
            # #     emd_memmap[i]=normalized_last_layer_states[i-count*batch_size]
            # #     paths_memmap[i]=i
            emd_memmap[count]
            count+=1