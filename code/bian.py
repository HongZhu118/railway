import json
import pandas as pd
import matplotlib.pyplot as plt

# 加载JSON数据
file_path = 'data.json'
with open(file_path, 'r', encoding='utf-8') as file:
    stations_data = json.load(file)
if '长沙西站' in stations_data:
    print(stations_data['长沙西站'])