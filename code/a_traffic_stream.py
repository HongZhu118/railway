import pandas as pd
import heapq
import numpy as np
from collections import deque, defaultdict

data = pd.read_excel('../G.xlsx')


station_list = []
# Populate the adjacency matrix based on the train routes
for _, group in data.groupby('车次'):
    stops = group['途径车站'].tolist()
    for i in range(len(stops) - 1):
        if stops[i] not in station_list:
            station_list.append(stops[i])

index_list = list(range(len(station_list)))
adj_matrix = pd.DataFrame(data=np.zeros((len(station_list), len(station_list)), dtype=int), index=index_list, columns=index_list)

for _, group in data.groupby('车次'):
    stops = group['途径车站'].tolist()
    for i in range(len(stops) - 1):
        index1 = station_list.index(stops[i])
        index2 = station_list.index(stops[i + 1])
        adj_matrix.at[index1, index2] += 1
        adj_matrix.at[index2, index1] += 1


print("站点总数：",len(adj_matrix))


def average_degree(adj_matrix):
    num_nodes = len(adj_matrix)
    total_degree = 0

    degree_dict = {}
    for i in range(num_nodes):
        degree = sum(adj_matrix[i])  # 求和得到节点 i 的度
        degree_dict[i] = degree
        total_degree += degree

    average_degree = total_degree / num_nodes
    print("total_degree:", total_degree)
    return average_degree,degree_dict
avg_degree,degree_dict = average_degree(adj_matrix)
print("Average degree:", avg_degree)
index_z = station_list.index('郑州东站')
row = adj_matrix[index_z]
list_ = [index for index, value in row.items() if value != 0]
list_1 = []
for i in list_:
    list_1.append(station_list[i])
print(list_1)
def compare(item):
    return item[1]  # 按值比较

# 使用max()函数，以比较函数为关键字参数
max_degree = max(degree_dict.items(),key=compare)
key = station_list[max_degree[0]]
print("station:",key,"max_degree:",max_degree[1])
def count_key_value_pairs_by_value(my_dict):
    value_pairs_count = {}  # 用于存储每个值对应的键值对个数

    for key, value in my_dict.items():
        if value in value_pairs_count:
            value_pairs_count[value] += 1
        else:
            value_pairs_count[value] = 1

    return value_pairs_count
result_dict = count_key_value_pairs_by_value(degree_dict)
print(result_dict)


def matrix_to_adj_list(matrix):
    adj_list = defaultdict(list)
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0 and matrix[i][j] != float('inf'):
                adj_list[i].append(j)
    return adj_list

# 转换为邻接表
graph= {}
adjacency_list = matrix_to_adj_list(adj_matrix)
for node, neighbors in adjacency_list.items():
    graph[node] = neighbors




def bfs(graph, start):
    """ 使用广度优先搜索计算从单一源到所有其他节点的最短路径 """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = deque([start])

    while queue:
        current_node = queue.popleft()
        current_distance = distances[current_node]

        for neighbor in graph[current_node]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

    return distances


def calculate_network_diameter_and_average_path_length(graph):
    all_distances = []
    for node in graph:
        distances = bfs(graph, node)
        all_distances.append(distances)

    max_distance = 0
    total_distance = 0
    count = 0

    nodes = list(graph.keys())
    for i in nodes:
        for j in nodes:
            if i != j:
                dist = all_distances[nodes.index(i)][j]
                if dist != float('inf'):
                    max_distance = max(max_distance, dist)
                    total_distance += dist
                    count += 1

    average_path_length = total_distance / count if count else float('inf')
    return max_distance, average_path_length,all_distances


network_diameter, average_path_length,all_distances = calculate_network_diameter_and_average_path_length(graph)
print(f"Network Diameter (D): {network_diameter}")
print(f"Average Path Length (L): {average_path_length}")


def clustering_coefficient(graph):
    """计算图中每个节点的聚类系数"""
    coefficients = {}
    for node in graph.keys():
        neighbors = list(graph[node])
        if len(neighbors) < 2:
            # 如果邻居少于2个，聚类系数为0，因为没有边可以形成
            coefficients[node] = 0
            continue

        # 计算邻居之间可能的连接数
        possible_links = len(neighbors) * (len(neighbors) - 1) / 2
        actual_links = 0

        # 计算实际存在的边数
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in graph[neighbors[i]]:
                    actual_links += 1

        # 计算聚类系数
        coefficients[node] = (2 * actual_links) / possible_links if possible_links > 0 else 0

    return coefficients


def average_clustering_coefficient(coefficients):
    """计算图的平均聚类系数"""
    return sum(coefficients.values()) / len(coefficients) if coefficients else 0

# 计算并打印每个节点的聚类系数
coeffs = clustering_coefficient(graph)
Clustering_dict = {}
for node, coeff in coeffs.items():
    Clustering_dict[node] = coeff
    # print(f"Node {node}: Clustering Coefficient = {coeff:.2f}")

# 计算并打印平均聚类系数
avg_coeff = average_clustering_coefficient(coeffs)
print(f"Average Clustering Coefficient = {avg_coeff:.2f}")


def degree_centrality(graph):
    # 度中心性：节点的邻居数除以N-1
    N = len(graph)
    dc = {node: len(neighbors) / (N - 1) for node, neighbors in graph.items()}
    return dc

def closeness_centrality(all_distances):
    # 接近中心性：(N-1)除以节点到所有其他节点的距离和
    N = len(all_distances)
    cc = {node: (N - 1) / sum(distances.values()) for node, distances in enumerate(all_distances) if sum(distances.values()) > 0}
    return cc

# 之前代码中计算所有节点对的最短路径
# distances = all_pairs_shortest_path(graph)

# 计算度中心性和接近中心性
dc = degree_centrality(graph)
cc = closeness_centrality(all_distances)

# 打印结果
print("Degree Centrality:", dc)
print("Closeness Centrality:", cc)