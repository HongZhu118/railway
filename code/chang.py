def swap_coordinates(coord):
    # coord 应该是一个形如 (longitude, latitude) 的元组
    return (str(coord[1]), str(coord[0]))

# 示例
original_coord = (
    102.065477, 25.191277
)
swapped_coord = swap_coordinates(original_coord)

print("Original Coordinates:", original_coord)
print( swapped_coord)
