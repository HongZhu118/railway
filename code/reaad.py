import json
import pandas as pd
import matplotlib.pyplot as plt
with open('stations.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Transform the dictionary, convert strings to floats and handle None values
station_coords = {station: (float(coords[1]), float(coords[0])) for station, coords in data.items() if coords}

for key,value in station_coords.items():
    if value[0]>100 and value[0]<103:
        if value[1]>25.5 and value[1]<28:
            print(key)