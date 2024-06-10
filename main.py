import os
import shutil
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Adam


# 加载JSON文件和图像路径
json_path = 'archive/groundtruth.json'
with open(json_path, 'r') as f:
    data = json.load(f)

dataset_path = 'archive/images'
output_path = 'archive/sorted_images'

# 创建输出文件夹


# 创建CNN模型
def create_cnn_model(input_shape):
    input_img = Input(shape=input_shape)

    # 第一个卷积层和池化层
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # 第二个卷积层和池化层
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # 第三个卷积层和池化层
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Flatten层和全连接层
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)

    # 输出层
    output = Dense(224 * 224 * 3, activation='sigmoid')(x)
    output = Reshape((224, 224, 3))(output)

    # 特征提取层（不进行分类）
    model = Model(inputs=input_img, outputs=output)

    return model

input_shape = (224, 224, 3)
model = create_cnn_model(input_shape)
model.summary()

# 数据增强
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1  # 10%的数据用于验证
)

train_generator = datagen.flow_from_directory(
    'archive/sorted_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='input',  # 对比学习使用输入图像本身作为标签
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'archive/sorted_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='input',
    subset='validation'
)


# 提取所有图像的特征
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data / 255.0  # 归一化
    return img_data


def extract_features(image_path, model):
    img_data = preprocess_image(image_path)
    features = model.predict(img_data)
    return features.flatten()


# 检索相似图像
def retrieve_similar_images(query_image_path, all_image_paths, features, model, top_k=3):
    query_features = extract_features(query_image_path, model)
    similarities = cosine_similarity([query_features], features)

    indices = np.argsort(similarities[0])[::-1][:top_k]
    similar_image_paths = [all_image_paths[idx] for idx in indices]

    return similar_image_paths


# 计算平均精度（AP）
def average_precision(retrieved_indices, relevant_indices):
    hits = 0
    sum_precisions = 0
    for i, idx in enumerate(retrieved_indices):
        if idx in relevant_indices:
            hits += 1
            sum_precisions += hits / (i + 1)
    if len(relevant_indices) == 0:
        return 0
    return sum_precisions / len(relevant_indices)


# 计算mAP
def calculate_map(query_image_paths, similar_image_paths, all_image_paths, features, model, top_k=3):
    aps = []
    for i, query_image_path in tqdm(enumerate(query_image_paths), desc="Calculating mAP"):
        relevant_indices = [all_image_paths.index(os.path.join(dataset_path, img)) for img in data[str(i)]['similar']]
        retrieved_image_paths = retrieve_similar_images(query_image_path, all_image_paths, features, model, top_k)
        retrieved_indices = [all_image_paths.index(img_path) for img_path in retrieved_image_paths]
        ap = average_precision(retrieved_indices, relevant_indices)
        aps.append(ap)
    return np.mean(aps)


# 初始化图像路径列表
query_image_paths = [os.path.join(dataset_path, data[key]['query']) for key in data]
similar_image_paths = [[os.path.join(dataset_path, img) for img in data[key]['similar']] for key in data]
optimizer = Adam(learning_rate=0.004)  # 使用Adam优化器并设置学习率
model.compile(optimizer=optimizer, loss='mse')
# 训练和评估循环
target_map = 0.7
current_map = 0
iteration = 0

while current_map < target_map:
    print(f"Training iteration {iteration + 1}")

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=5
    )

    # 提取特征
    all_image_paths = query_image_paths + sum(similar_image_paths, [])
    features = []

    for img_path in tqdm(all_image_paths, desc="Extracting Features"):
        features.append(extract_features(img_path, model))

    features = np.array(features)

    # 计算mAP
    current_map = calculate_map(query_image_paths, similar_image_paths, all_image_paths, features, model, top_k=3)
    print(f"Mean Average Precision (mAP): {current_map}")

    iteration += 1


# 显示查询图像和检索到的相似图像
def show_images(image_paths, cols=1):
    plt.figure(figsize=(20, 10))
    for i, img_path in enumerate(image_paths):
        img = image.load_img(img_path)
        plt.subplot(len(image_paths) // cols + 1, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


# 示例：展示一个查询图像和前3个相似图像
query_image_path = query_image_paths[0]
similar_images = retrieve_similar_images(query_image_path, all_image_paths, features, model, top_k=3)

print("Query Image:")
show_images([query_image_path])
print("Similar Images:")
show_images(similar_images)
