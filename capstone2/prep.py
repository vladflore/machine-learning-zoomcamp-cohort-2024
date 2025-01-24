import os
import matplotlib.pyplot as plt
import shutil
import random
from PIL import Image
import re

IMAGES_DATASET = "content/drive/mydrive/mlzoomcamp/capstone2/materials"

materials_folders = os.listdir(IMAGES_DATASET)

for folder_name in materials_folders:
  new_folder_name = re.sub(r"[ ]+", "_", folder_name.lower())
  if folder_name != new_folder_name:
    os.rename(os.path.join(IMAGES_DATASET, folder_name), os.path.join(IMAGES_DATASET, new_folder_name))
materials_folders = os.listdir(IMAGES_DATASET)
print(materials_folders)

image_count = 0

for material_folder in materials_folders:
    material_path = os.path.join(IMAGES_DATASET, material_folder)
    if os.path.isdir(material_path):
        images = os.listdir(material_path)
        image_count += len(images)

print(f"Total number of images in the dataset: {image_count}")

image_counts = {}
for material_folder in materials_folders:
    material_path = os.path.join(IMAGES_DATASET, material_folder)
    if os.path.isdir(material_path):
        image_count = len([f for f in os.listdir(material_path)])
        image_counts[material_folder] = image_count

total_images = sum(image_counts.values())
print(f"Total number of usable images in the dataset: {total_images}")

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for material_folder in os.listdir(source_dir):
        material_source_path = os.path.join(source_dir, material_folder)
        if os.path.isdir(material_source_path):
            material_train_path = os.path.join(train_dir, material_folder)
            os.makedirs(material_train_path, exist_ok=True)
            material_val_path = os.path.join(val_dir, material_folder)
            os.makedirs(material_val_path, exist_ok=True)
            material_test_path = os.path.join(test_dir, material_folder)
            os.makedirs(material_test_path, exist_ok=True)

            images = [f for f in os.listdir(material_source_path) if os.path.isfile(os.path.join(material_source_path, f))]
            random.shuffle(images)

            train_split = int(len(images) * train_ratio)
            val_split = int(len(images) * (train_ratio + val_ratio))

            for i, image in enumerate(images):
              source_path = os.path.join(material_source_path, image)
              if i < train_split:
                  destination_path = os.path.join(material_train_path, image)
              elif i < val_split:
                  destination_path = os.path.join(material_val_path, image)
              else:
                  destination_path = os.path.join(material_test_path, image)
              shutil.copy2(source_path, destination_path)

train_dir = f'./content/drive/mydrive/mlzoomcamp/capstone2/dataset/train/materials'
val_dir = f'./content/drive/mydrive/mlzoomcamp/capstone2/dataset/val/materials'
test_dir = f'./content/drive/mydrive/mlzoomcamp/capstone2/dataset/test/materials'

split_dataset(IMAGES_DATASET, train_dir, val_dir, test_dir)

total_train, total_val, total_test = 0, 0, 0
for material_folder in materials_folders:
  dir_path = os.path.join(train_dir, material_folder)
  train_images = len(os.listdir(dir_path))
  dir_path = os.path.join(val_dir, material_folder)
  val_images = len(os.listdir(dir_path))
  dir_path = os.path.join(test_dir, material_folder)
  test_images = len(os.listdir(dir_path))

  total_train += train_images
  total_val += val_images
  total_test += test_images

  print(f'{material_folder}: train={train_images}, val={val_images}, test={test_images}')
  assert train_images + val_images + test_images == image_counts[material_folder]

print(f'Total train images: {total_train}')
print(f'Total val images: {total_val}')
print(f'Total test images: {total_test}')