import os
import pickle
import numpy as np

def read_and_output_files(save_folder):
    # 确定文件路径
    kmeans_index_file = os.path.join(save_folder, 'kmeans_index.pickle')
    centroids_file = os.path.join(save_folder, 'kmeans_centroids.npy')
    dist_to_cent_file = os.path.join(save_folder, 'dist_to_cent.npy')
    nearest_cent_file = os.path.join(save_folder, 'nearest_cent.npy')

    # 读取并输出 faiss K-means 索引对象
    if os.path.exists(kmeans_index_file):
        with open(kmeans_index_file, 'rb') as file:
            kmeans_index = pickle.load(file)
        print("K-means Index Object:")
        print(kmeans_index)
    else:
        print("K-means Index file not found.")

    # 读取并输出 K-means 聚类中心
    if os.path.exists(centroids_file):
        centroids = np.load(centroids_file)
        print("\nK-means Centroids:")
        print(centroids.shape)
    else:
        print("K-means Centroids file not found.")

    # 读取并输出数据点到聚类中心的距离
    if os.path.exists(dist_to_cent_file):
        dist_to_cent = np.load(dist_to_cent_file)
        print("\nDistance to Centroid for each data point:")
        print(dist_to_cent.shape)
    else:
        print("Distance to Centroid file not found.")

    # 读取并输出数据点的最近聚类中心
    if os.path.exists(nearest_cent_file):
        nearest_cent = np.load(nearest_cent_file)
        print("\nNearest Centroid for each data point:")
        print(nearest_cent.shape)
    else:
        print("Nearest Centroid file not found.")

# 设置保存文件的路径
save_folder = '/home/guochuanzhe/data-process/SemDeDup/memory/save_folder'

# 调用函数
read_and_output_files(save_folder)
