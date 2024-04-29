import pandas as pd
import heapq
import numpy as np
from collections import deque, defaultdict

# data1 = pd.read_excel('../files/1.xlsx')
# data2 = pd.read_excel('../files/2.xlsx')
# data3 = pd.read_excel('../files/3.xlsx')
# data4 = pd.read_excel('../files/倒数第二行.xlsx')
data5 = pd.read_excel('../files/副本1(1).xlsx')
data = pd.read_csv('../files/route.csv')
# data_ = pd.read_excel('../files/5.xlsx')
# list_ = data_['Hsrwsnm'].tolist()


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


# for _ , group in data1.groupby('字段2_文本'):
#     stops = group['字段1_文本'].tolist()
#     for i in range(len(stops) - 1):
#         index1 = station_list.index(stops[i])
#         index2 = station_list.index(stops[i + 1])
#         adj_matrix.at[index1, index2] = 1
#         adj_matrix.at[index2, index1] = 1
#
# for _ ,group in data2.groupby('字段2_文本'):
#     stops = group['字段1_文本'].tolist()
#     for i in range(len(stops) - 1):
#         index1 = station_list.index(stops[i])
#         index2 = station_list.index(stops[i + 1])
#         adj_matrix.at[index1, index2] = 1
#         adj_matrix.at[index2, index1] = 1
#
# for _ ,group in data3.groupby('文本'):
#     stops = group['字段1_文本'].tolist()
#     for i in range(len(stops) - 1):
#         index1 = station_list.index(stops[i])
#         index2 = station_list.index(stops[i + 1])
#         adj_matrix.at[index1, index2] = 1
#         adj_matrix.at[index2, index1] = 1
#
# for _ ,group in data4.groupby('文本'):
#     stops = group['字段1_文本'].tolist()
#     for i in range(len(stops) - 1):
#         index1 = station_list.index(stops[i])
#         index2 = station_list.index(stops[i + 1])
#         adj_matrix.at[index1, index2] = 1
#         adj_matrix.at[index2, index1] = 1

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

# new_station_list = []
# for i in station_list:
#     if i[:-1] not in list_:
#         index = station_list.index(i)
#         adj_matrix = adj_matrix.drop(index,axis=0)
#         adj_matrix = adj_matrix.drop(index,axis=1)
#     else:
#         new_station_list.append(i)

# new_index_list = list(range(len(station_list)))
# new_adj_matrix = pd.DataFrame(data=np.zeros((len(new_station_list), len(new_station_list)), dtype=int), index=new_index_list, columns=new_index_list)
# 使用列表收集所有行
# rows = []
# for index, row in adj_matrix.iterrows():
#     rows.append(row.tolist())

# 创建新的DataFrame
# new_adj_matrix = pd.DataFrame(rows, columns=adj_matrix.columns)
# adj_matrix = adj_matrix.reset_index(drop=True)
# adj_matrix.columns = range(adj_matrix.shape[1])
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


# def compute_betweenness_centrality(graph, all_distances):
#     betweenness = defaultdict(float)
#     vertices = list(graph.keys())
#     for start in vertices:
#         # Initialize dictionaries for path count and predecessors along the shortest path
#         path_count = {v: 0 for v in vertices}  # Number of shortest paths through node
#         path_count[start] = 1
#         predecessors = {v: [] for v in vertices}  # Predecessors in shortest paths
#
#         # Use a BFS to find shortest paths from start node
#         queue = deque([start])
#         while queue:
#             current = queue.popleft()
#             for neighbor in graph[current]:
#                 # Only proceed if it's a shortest path
#                 if all_distances[start][neighbor] == all_distances[start][current] + 1:
#                     if not predecessors[neighbor]:
#                         queue.append(neighbor)
#                     path_count[neighbor] += path_count[current]
#                     predecessors[neighbor].append(current)
#
#         # Accumulate betweenness values for nodes in shortest paths
#         contributions = {v: 0 for v in vertices}
#         for node in vertices[::-1]:
#             if node != start:
#                 for pred in predecessors[node]:
#                     contrib = (1 + contributions[node]) * (path_count[pred] / path_count[node])
#                     contributions[pred] += contrib
#                     betweenness[pred] += contrib / 2  # divide by 2 for undirected graph
#
#     # Normalize the betweenness values
#     for node in betweenness:
#         betweenness[node] /= ((len(vertices) - 1) * (len(vertices) - 2) / 2)
#
#     return dict(betweenness)
#
#
# # Given the adjacency list and all_distances from your previous functions:
# betweenness_centrality = compute_betweenness_centrality(graph, all_distances)
# print("Betweenness Centrality:")
# for node, centrality in betweenness_centrality.items():
#     print(f"Node {node}: {centrality}")
