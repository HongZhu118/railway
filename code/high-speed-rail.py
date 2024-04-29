import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
# 设置 Matplotlib 支持中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows下使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 1. 数据加载
stations_df = pd.read_csv('../railway stations delay data.csv')
adjacent_df = pd.read_csv('../adjacent railway stations mileage data.csv')
trains_operation_df = pd.read_csv('../high-speed trains operation data.csv')
junctions_df = pd.read_csv('../junction stations data.csv')

# 2. 网络构建
G = nx.Graph()
for index, row in adjacent_df.iterrows():
    G.add_edge(row['from_station'], row['to_station'], weight=row['mileage'])

# 添加站点属性
for index, row in stations_df.iterrows():
    if row['station_name'] in G:
        G.nodes[row['station_name']]['station_type'] = row['station_type']
        # 其他属性可按需添加

# 3. 网络分析
degrees = nx.degree(G)
print("平均节点度:", np.mean(np.array([degree for node, degree in degrees])))
#移除边的函数
def remove_edge(G, method='random'):
    if method == 'random':
        # Directly use Python's random.choice on a list of edges
        edge_to_remove = random.choice(list(G.edges()))
    elif method == 'highest_betweenness':
        edge_betweenness = nx.edge_betweenness_centrality(G)
        edge_to_remove = max(edge_betweenness, key=edge_betweenness.get)
    G.remove_edge(*edge_to_remove)  # Use tuple unpacking to remove the edge
    return G
# 4. 脆弱性模拟
def remove_node(G, method='random'):
    if method == 'random':
        node_to_remove = np.random.choice(G.nodes())
    elif method == 'highest_degree':
        node_to_remove = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
    # 其他方法可按需添加
    G.remove_node(node_to_remove)
    return G


# 移除节点或连边并观察平均路径长度的变化
def simulate_attacks_with_avg_path_length(G, attack_type='node', method='random', iterations=10):
    avg_path_length_list = []
    for _ in range(iterations):
        if attack_type == 'node':
            G = remove_node(G, method=method)
        elif attack_type == 'edge':
            G = remove_edge(G, method=method)
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            avg_path_length_list.append(avg_path_length)
        else:
            # Handle the case for disconnected graph
            avg_path_length_list.append(float('inf'))  # or np.nan, depending on how you want to handle this
    return avg_path_length_list
# 连边攻击的模拟函数
def simulate_edge_attacks_with_avg_path_length(G, method='random', iterations=10):
    avg_path_length_list = []
    for _ in range(iterations):
        G = remove_edge(G, method=method)
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            avg_path_length_list.append(avg_path_length)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(largest_cc) > 1:  # Make sure there's more than one node in the subgraph
                avg_path_length = nx.average_shortest_path_length(subgraph)
                avg_path_length_list.append(avg_path_length)
            else:
                avg_path_length_list.append(float('inf'))  # or np.nan
    return avg_path_length_list

# 执行节点和连边攻击的模拟
iterations = 100  # 迭代次数
random_node_attack_avg_path_length = simulate_attacks_with_avg_path_length(G.copy(), 'node', 'random', iterations)
targeted_node_attack_avg_path_length = simulate_attacks_with_avg_path_length(G.copy(), 'node', 'highest_degree', iterations)
random_edge_attack_avg_path_length = simulate_edge_attacks_with_avg_path_length(G.copy(), 'random', iterations)
targeted_edge_attack_avg_path_length = simulate_edge_attacks_with_avg_path_length(G.copy(), 'highest_betweenness', iterations)

# 可视化节点攻击的平均路径长度结果
plt.figure(figsize=(14, 7))

# 节点攻击图
plt.subplot(1, 2, 1)
fractions_node = np.linspace(0, 1, len(random_node_attack_avg_path_length))
plt.plot(fractions_node, random_node_attack_avg_path_length, 'k-', marker='o', label='随机节点攻击效应')
plt.plot(fractions_node, targeted_node_attack_avg_path_length, 'r-', marker='x', label='蓄意节点攻击效应')
plt.xlabel('攻击节点比例')
plt.ylabel('平均路径长度')
plt.title('节点攻击效应图')
plt.legend(loc='upper right')
plt.grid(True)

# 连边攻击图
plt.subplot(1, 2, 2)
fractions_edge = np.linspace(0, 1, len(random_edge_attack_avg_path_length))
plt.plot(fractions_edge, random_edge_attack_avg_path_length, 'b-', marker='o', label='随机连边攻击效应')
plt.plot(fractions_edge, targeted_edge_attack_avg_path_length, 'b--', marker='x', label='蓄意连边攻击效应')
plt.xlabel('攻击连边比例')
plt.ylabel('平均路径长度')
plt.title('连边攻击效应图')
plt.legend(loc='upper right')
plt.grid(True)

# 调整子图布局
plt.tight_layout()
plt.show()
# 从文件中加载数据
stations_delay_df = pd.read_csv('../railway stations delay data.csv')

# 假设 `start_time` 包含了可以被转换为日期的时间戳
stations_delay_df['date'] = pd.to_datetime(stations_delay_df['start_time']).dt.date

# 接下来根据车站名称和日期来聚合延误数，这里我们只计算出发的延误
# 可能还需要根据实际数据调整列名，以下假设车站名称为 'station_name'
nanjingnan_delays = stations_delay_df[stations_delay_df['station_name'] == 'Nanjingnan Railway Station'].groupby('date')['up_departure_delay_number'].sum()
hangzhou_delays = stations_delay_df[stations_delay_df['station_name'] == 'Hangzhou Railway Station'].groupby('date')['up_departure_delay_number'].sum()
shanghaihongqiao_delays = stations_delay_df[stations_delay_df['station_name'] == 'Shanghaihongqiao Railway Station'].groupby('date')['up_departure_delay_number'].sum()



# 绘制延误数随时间变化的图表
plt.figure(figsize=(10, 5))
plt.plot(nanjingnan_delays.index, nanjingnan_delays.values, label='南京南站')
plt.plot(hangzhou_delays.index, hangzhou_delays.values, label='杭州站')
plt.plot(shanghaihongqiao_delays.index, shanghaihongqiao_delays.values, label='上海虹桥站')

# 添加标签和标题
plt.xlabel('日期')
plt.ylabel('延误数')
plt.title('三个车站的延误数随时间变化')
plt.legend()
plt.tight_layout()
plt.show()

# 计算介数中心性和归一化
betweenness_dict = nx.betweenness_centrality(G)  # 返回字典
betweenness_norm = np.array(list(betweenness_dict.values()))
betweenness_norm /= betweenness_norm.max()  # 归一化

# 计算度中心性
degree_dict = nx.degree_centrality(G)
degree_norm = np.array(list(degree_dict.values()))
degree_norm /= degree_norm.max()  # 归一化

# 计算接近中心性
closeness_dict = nx.closeness_centrality(G)
closeness_norm = np.array(list(closeness_dict.values()))
closeness_norm /= closeness_norm.max()  # 归一化

# 计算特征向量中心性
eigenvector_dict = nx.eigenvector_centrality(G, max_iter=500)
eigenvector_norm = np.array(list(eigenvector_dict.values()))
eigenvector_norm /= eigenvector_norm.max()  # 归一化

# 创建节点颜色数组
node_color = closeness_norm  # 使用接近中心性作为颜色

# 创建节点大小数组
node_size = 2000 * betweenness_norm  # 使用介数中心性作为大小

# 创建节点边框粗细数组，这里为了可视化我们放大特征向量中心性
node_border = 5.0 * eigenvector_norm

# 设置图的布局
pos = nx.spring_layout(G)

# 绘制网络
plt.figure(figsize=(15, 15))
nx.draw_networkx_edges(G, pos, alpha=0.2)
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_size,
    node_color=node_color,
    cmap=plt.cm.plasma,
    edgecolors='black',  # 节点边框颜色
    linewidths=node_border  # 节点边框宽度
)
nx.draw_networkx_labels(G, pos, font_color='white')
plt.title('Network Visualization with Multiple Centrality Indicators')
plt.colorbar(nodes)
plt.axis('off')
plt.show()

