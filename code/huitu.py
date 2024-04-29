import pandas as pd
import requests
import json

# 路径可能需要根据你的文件存放位置进行调整
route_file_path = '../files/route.csv'

# 读取CSV文件
route_data = pd.read_csv(route_file_path, encoding='utf-8')  # 确保使用utf-8编码读取CSV

# 解析站点列表，假设每个路线以'、'分隔站点
route_list = [route.split('、') for route in route_data['Mnst']]

def get_coordinates(place):
    url = 'https://nominatim.openstreetmap.org/search'
    params = {
        'q': place,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data:
        latitude = data[0]['lat']
        longitude = data[0]['lon']
        return (latitude, longitude)
    else:
        return None

def add_to_json(file_path, key, value):
    try:
        # 尝试打开现有的json文件
        with open(file_path, 'r', encoding='utf-8') as file:  # 使用utf-8编码打开
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或文件为空，则初始化一个空字典
        data = {}

    # 更新数据，添加新的键值对
    data[key] = value

    # 将更新后的数据写回文件
    with open(file_path, 'w', encoding='utf-8') as file:  # 使用utf-8编码写入
        json.dump(data, file, indent=4)  # 使用indent参数美化输出

# 初始化数据存储文件
output_file_path = 'data.json'

# 遍历每个路线中的站点
for station_list in route_list[1:]:
    for station in station_list:
        n_station = station + '站'
        coordinates = get_coordinates(n_station)
        print(f"{n_station}: {coordinates}")
        add_to_json(output_file_path, n_station, coordinates)
