import pandas as pd
import heapq
import numpy as np
from collections import deque, defaultdict

import networkx as nx

data5 = pd.read_excel('../files/副本1(1).xlsx')
data = pd.read_csv('../files/route.csv')


station_list = []

for _ ,group in data5.groupby('路1'):
    stops = group['站1'].tolist()
    for i in range(len(stops)):
        if stops[i] not in station_list:
            station_list.append(stops[i])

for _ ,group in data5.groupby('路2'):
    stops = group['站2'].tolist()
    for i in range(len(stops)):
        if stops[i] not in station_list:
            station_list.append(stops[i])

for _ ,group in data5.groupby('路3'):
    stops = group['站3'].tolist()
    for i in range(len(stops)):
        if stops[i] not in station_list:
            station_list.append(stops[i])


# 提取指定列到列表
column_data = data['Mnst'].tolist()
# 打印提取的列数据
new_column_data = column_data[1:]
data = {"Mnst": new_column_data}
df = pd.DataFrame(data)
for line in df['Mnst']:
    stations = line.split('、')
    for _ in stations:
        if _ +'站' not in station_list:
            station_list.append(_ + '站')

index_list = list(range(len(station_list)))
adj_matrix = pd.DataFrame(data=np.zeros((len(station_list), len(station_list)), dtype=int), index=index_list, columns=index_list)

for _ ,group in data5.groupby('路1'):
    stops = group['站1'].tolist()
    for i in range(len(stops) - 1):
        index1 = station_list.index(stops[i])
        index2 = station_list.index(stops[i + 1])
        adj_matrix.at[index1, index2] = 1
        adj_matrix.at[index2, index1] = 1
for _ ,group in data5.groupby('路2'):
    stops = group['站2'].tolist()
    for i in range(len(stops) - 1):
        index1 = station_list.index(stops[i])
        index2 = station_list.index(stops[i + 1])
        adj_matrix.at[index1, index2] = 1
        adj_matrix.at[index2, index1] = 1
for _ ,group in data5.groupby('路3'):
    stops = group['站3'].tolist()
    for i in range(len(stops) - 1):
        index1 = station_list.index(stops[i])
        index2 = station_list.index(stops[i + 1])
        adj_matrix.at[index1, index2] = 1
        adj_matrix.at[index2, index1] = 1
for line in df['Mnst']:
    stops = line.split('、')
    for i in range(len(stops) - 1):
        index1 = station_list.index(stops[i] + '站')
        index2 = station_list.index(stops[i + 1] + '站')
        adj_matrix.at[index1, index2] = 1
        adj_matrix.at[index2, index1] = 1

print("站点总数：",len(adj_matrix))


# 创建图对象
G = nx.from_numpy_array(adj_matrix)

# 网络平均度
average_degree = np.mean([deg for node, deg in G.degree()])

# 网络直径（需要确保图是连通的）
if nx.is_connected(G):
    diameter = nx.diameter(G)
else:
    diameter = float('inf')  # 如果图不连通，无法定义直径

# 平均路径长度
average_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')

# 节点的聚类系数
clustering_coefficients = nx.clustering(G)

# 整个网络的平均聚类系数
average_clustering_coefficient = nx.average_clustering(G)

# 度中心性
degree_centrality = nx.degree_centrality(G)

# 特征向量中心性
eigenvector_centrality = nx.eigenvector_centrality(G)

# 介数中心性
betweenness_centrality = nx.betweenness_centrality(G)

# 打印结果
print("Average Degree:", average_degree)
print("Diameter:", diameter)
print("Average Path Length:", average_path_length)
print("Clustering Coefficients:", clustering_coefficients)
print("Average Clustering Coefficient:", average_clustering_coefficient)
print("Degree Centrality:", degree_centrality)
print("Eigenvector Centrality:", eigenvector_centrality)
print("Betweenness Centrality:", betweenness_centrality)
