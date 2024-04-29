import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the JSON file
with open('stations.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Transform the dictionary, convert strings to floats and handle None values
station_coords = {station: (float(coords[1]), float(coords[0])) for station, coords in data.items() if coords}


# Load the route data from the CSV file
route_file_path = '../files/route.csv'
route_data = pd.read_csv(route_file_path)
route_list = [route.split('、') for route in route_data['Mnst']]

# Function to plot routes based on station names
def plot_route(station_list):
    route_lons = []
    route_lats = []
    for station in station_list:
        new_station = station + '站'
        if new_station in station_coords:
            coord = station_coords[new_station]
            route_lons.append(coord[1])
            route_lats.append(coord[0])
    return route_lons, route_lats

# Plot each route
plt.figure(figsize=(12, 10))
for stations in route_list:
    route_lons, route_lats = plot_route(stations)
    if route_lons and route_lats:
        plt.plot( route_lats,route_lons, marker='o', color='black')



# Adding map details
plt.title('Station Routes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()
