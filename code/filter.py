import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import json
import pandas as pd
import numpy as np

data_G = pd.read_excel('../G.xlsx')
data_L = pd.read_csv('../files/route.csv')

# Extract unique stations
stations_G = data_G['途径车站'].unique()
column_data = data_L['Mnst'].tolist()
# 打印提取的列数据
new_column_data = column_data[1:]

data = {
    "Mnst": new_column_data
}

df = pd.DataFrame(data)

# 解析站点并构建唯一站点列表
stations_L = set()
for line in df['Mnst']:
    stations = line.split('、')
    stations_L.update(stations)
station_L_list = list(stations_L)
station_list = []
for i in range(len(station_L_list)):
    str_to_check = station_L_list[i]+'站'
    is_in_array = np.any(stations_G == str_to_check)
    if is_in_array:
        station_list.append(station_L_list[i])
print(len(station_list))