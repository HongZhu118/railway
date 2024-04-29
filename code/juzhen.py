import numpy as np
from collections import deque, defaultdict
def create_adjacency_matrix(train_stations_filename, additional_stations_filename, filter_stations_filename):
    stations = []
    connections = []
    line_dict = {}

    # Read the first TXT file
    with open(train_stations_filename, 'r', encoding='utf-8') as file:
        for line in file:
            stations_in_line = line.strip().split('、')
            for station in stations_in_line:
                if station not in stations:
                    stations.append(station)
            connections.extend([(stations_in_line[i], stations_in_line[i+1]) for i in range(len(stations_in_line) - 1)])

    # Read the second TXT file and group stations by line
    with open(additional_stations_filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            station_name = parts[0].replace('站', '')
            line_name = parts[1]
            if station_name not in stations:
                stations.append(station_name)
            if line_name not in line_dict:
                line_dict[line_name] = []
            line_dict[line_name].append(station_name)

    # Add connections based on line order from the second file
    for line_stations in line_dict.values():
        for i in range(len(line_stations) - 1):
            connections.append((line_stations[i], line_stations[i + 1]))

    # Read the third TXT file for valid stations
    valid_stations = set()
    with open(filter_stations_filename, 'r', encoding='utf-8') as file:
        for line in file:
            station_name = line.strip()
            valid_stations.add(station_name)

    # Filter stations and connections
    stations = [station for station in stations if station in valid_stations]
    filtered_connections = [(s1, s2) for s1, s2 in connections if s1 in valid_stations and s2 in valid_stations]

    # Create the adjacency matrix
    n = len(stations)
    adj_matrix = np.zeros((n, n), dtype=int)
    for s1, s2 in filtered_connections:
        idx1 = stations.index(s1)
        idx2 = stations.index(s2)
        adj_matrix[idx1, idx2] = 1
        adj_matrix[idx2, idx1] = 1

    return stations, adj_matrix

# Provide paths to the TXT files
train_stations_filename = '../files/chezhan(2).txt'
additional_stations_filename = '../files/zhanci.txt'
filter_stations_filename = '../files/all.txt'

# Generate stations and adjacency matrix
station_list, adj_matrix = create_adjacency_matrix(train_stations_filename, additional_stations_filename, filter_stations_filename)

# # Save the stations order to a file
# with open('C:\\Users\\yee\\pythonProject1\\train\\StationOrder.txt', 'w', encoding='utf-8') as file:
#     file.write('\n'.join(stations))
#
# # Save the adjacency matrix with headers
# header = ' ' + ' '.join(stations)  # First space for leading corner
# np.savetxt('C:\\Users\\yee\\pythonProject1\\train\\Matrix.txt', adj_matrix, fmt='%d', delimiter=' ', header=header, comments='')
#
# # Additional output and analysis as before
# print("总节点数:", len(stations))
# degrees = adj_matrix.sum(axis=0)
# top_ten_indices = np.argsort(-degrees)[:10]
# top_ten_stations = [(stations[i], degrees[i]) for i in top_ten_indices]
# print("度数最高的前十个节点:")
# for station, degree in top_ten_stations:
#     print(station, degree)
# degree_count = np.zeros(17, dtype=int)
# for degree in degrees:
#     if 1 <= degree <= 16:
#         degree_count[degree] += 1
# print("各度数的节点数（1到16）:")
# for i in range(1, 17):
#     print(f"度数 {i}: {degree_count[i]}")

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