import os
import shutil
import json

# 加载JSON文件和图像路径
json_path = 'archive/groundtruth.json'
with open(json_path, 'r') as f:
    data = json.load(f)

dataset_path = 'archive/images'
output_path = 'archive/sorted_images'

# 创建输出文件夹
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 按类别组织图像
for key in data:
    query_image_path = os.path.join(dataset_path, data[key]['query'])
    similar_image_paths = [os.path.join(dataset_path, img) for img in data[key]['similar']]

    # 创建类别文件夹
    query_class_folder = os.path.join(output_path, f'class_{key}_query')
    similar_class_folder = os.path.join(output_path, f'class_{key}_similar')

    if not os.path.exists(query_class_folder):
        os.makedirs(query_class_folder)
    if not os.path.exists(similar_class_folder):
        os.makedirs(similar_class_folder)

    # 移动图像到类别文件夹
    shutil.copy(query_image_path, query_class_folder)
    for img_path in similar_image_paths:
        shutil.copy(img_path, similar_class_folder)

print("Images have been organized into class folders.")
