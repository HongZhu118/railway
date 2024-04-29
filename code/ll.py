import json


def txt_to_json(input_file, output_file):
    station_coords = {}

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            # 确保行包含冒号分隔的两部分
            parts = line.strip().split(':')
            if len(parts) == 2:
                station = parts[0].strip()
                # 使用 try-except 来捕获解析错误
                try:
                    # 尝试解析坐标
                    coords = eval(parts[1].strip())
                    if coords and len(coords) == 2:  # 确保坐标是个包含两个元素的元组
                        station_coords[station] = [str(coords[0]), str(coords[1])]
                except SyntaxError as e:
                    print(f"Error parsing coordinates for {station}: {e}")
                except TypeError as e:
                    print(f"Invalid data format for {station}: {e}")

    # 将字典写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(station_coords, json_file, ensure_ascii=False, indent=4)


# 指定输入文件和输出文件的路径
input_file = '../files/坐标(4).txt'
output_file = 'stations.json'

# 调用函数转换文件
txt_to_json(input_file, output_file)
