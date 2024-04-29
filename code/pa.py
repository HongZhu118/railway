import pandas as pd
import requests
import json
# 路径可能需要根据你的文件存放位置进行调整
route_file_path = '../files/route.csv'

# 读取CSV文件
route_data = pd.read_csv(route_file_path)

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
# 创建一个集合用来存储所有独特的站点
# unique_stations = set()
dict = {}
# 遍历每个路线中的站点
for station_list in route_list[1:]:
    for station in station_list:
        n_station = station + '站'
        coordinates = get_coordinates(n_station)
        dict[n_station] = coordinates
        print(n_station+":",coordinates)

print(dict)
json_data = json.dumps(dict)
with open('data.json', 'w') as file:
    file.write(json_data)