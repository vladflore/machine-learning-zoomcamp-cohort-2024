import os
import matplotlib.pyplot as plt
import shutil
import random
from PIL import Image

ROOT = "content/drive/mydrive/mlzoomcamp/capstone1/animals"

IMAGES_DATASET = f"{ROOT}/raw-img"

TO_SKIP = "to_skip.txt"

train_dir = f"{ROOT}/train"
val_dir = f"{ROOT}/val"
test_dir = f"{ROOT}/test"

train_dir_small = f"{ROOT}/small/train"
val_dir_small = f"{ROOT}/small/val"
test_dir_small = f"{ROOT}/small/test"


def count_images():
    image_count = 0
    for animal_folder in os.listdir(IMAGES_DATASET):
        animal_path = os.path.join(IMAGES_DATASET, animal_folder)
        if os.path.isdir(animal_path):
            images = os.listdir(animal_path)
            image_count += len(images)

    print(f"Total number of images in the dataset: {image_count}")


def find_invalid_images():
    to_skip = []
    for animal_folder in os.listdir(IMAGES_DATASET):
        folder_path = os.path.join(IMAGES_DATASET, animal_folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as f:
                is_jfif = b"JFIF" in f.peek(10)
                if not is_jfif:
                    to_skip.append(file_name)
                    print(f"{file_path} is not a valid JPEG file")

    print(f"Total number of files to skip: {len(to_skip)}")

    print("Writing invalid files to to_skip.txt")
    with open(TO_SKIP, "w") as f:
        for item in to_skip:
            f.write("%s\n" % item)


def load_invalid_images():
    to_skip = []
    with open(TO_SKIP, "r") as f:
        for line in f:
            to_skip.append(line.strip())
    print(f"Total number of files to skip: {len(to_skip)}")
    return to_skip


def count_images_per_animal():
    image_counts = {}
    to_skip = load_invalid_images()
    for animal_folder in os.listdir(IMAGES_DATASET):
        animal_path = os.path.join(IMAGES_DATASET, animal_folder)
        if os.path.isdir(animal_path):
            image_count = len([f for f in os.listdir(animal_path) if f not in to_skip])
            image_counts[animal_folder] = image_count

    total_images = sum(image_counts.values())
    print(f"Total number of usable images in the dataset: {total_images}")
    return image_counts


def plot_histo(image_counts):
    print("Plotting histogram of number of images per animal folder")
    plt.figure(figsize=(10, 6))
    plt.bar(image_counts.keys(), image_counts.values())
    plt.xlabel("Animal Folders")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Animal Folder")
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(image_counts.values()):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def split_dataset(
    source_dir,
    train_dir,
    val_dir,
    test_dir,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
):
    print("Splitting dataset into train, val, and test sets")

    to_skip = load_invalid_images()
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for animal_folder in os.listdir(source_dir):
        animal_source_path = os.path.join(source_dir, animal_folder)
        if os.path.isdir(animal_source_path):
            animal_train_path = os.path.join(train_dir, animal_folder)
            os.makedirs(animal_train_path, exist_ok=True)
            animal_val_path = os.path.join(val_dir, animal_folder)
            os.makedirs(animal_val_path, exist_ok=True)
            animal_test_path = os.path.join(test_dir, animal_folder)
            os.makedirs(animal_test_path, exist_ok=True)

            images = [
                f
                for f in os.listdir(animal_source_path)
                if f not in to_skip
                and os.path.isfile(os.path.join(animal_source_path, f))
            ]
            random.shuffle(images)

            train_split = int(len(images) * train_ratio)
            val_split = int(len(images) * (train_ratio + val_ratio))

            for i, image in enumerate(images):
                source_path = os.path.join(animal_source_path, image)
                if i < train_split:
                    destination_path = os.path.join(animal_train_path, image)
                elif i < val_split:
                    destination_path = os.path.join(animal_val_path, image)
                else:
                    destination_path = os.path.join(animal_test_path, image)
                shutil.copy2(source_path, destination_path)


def check_split():
    print("Checking the split of train, val, and test sets")
    total_train, total_val, total_test = 0, 0, 0
    image_counts = count_images_per_animal()

    for animal_folder in os.listdir(IMAGES_DATASET):
        dir_path = os.path.join(train_dir, animal_folder)
        train_images = len(os.listdir(dir_path))
        dir_path = os.path.join(val_dir, animal_folder)
        val_images = len(os.listdir(dir_path))
        dir_path = os.path.join(test_dir, animal_folder)
        test_images = len(os.listdir(dir_path))

        total_train += train_images
        total_val += val_images
        total_test += test_images

        print(
            f"{animal_folder}: train={train_images}, val={val_images}, test={test_images}"
        )
        assert train_images + val_images + test_images == image_counts[animal_folder]

    print(f"Total train images: {total_train}")
    print(f"Total val images: {total_val}")
    print(f"Total test images: {total_test}")


def copy_subset_of_images(source_dir, dest_dir, percentage):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for animal_folder in os.listdir(source_dir):
        animal_source_path = os.path.join(source_dir, animal_folder)
        if os.path.isdir(animal_source_path):
            animal_dest_path = os.path.join(dest_dir, animal_folder)
            os.makedirs(animal_dest_path, exist_ok=True)

            images = [
                f
                for f in os.listdir(animal_source_path)
                if os.path.isfile(os.path.join(animal_source_path, f))
            ]
            num_images_to_copy = int(len(images) * percentage)
            images_to_copy = random.sample(images, num_images_to_copy)

            for image in images_to_copy:
                source_path = os.path.join(animal_source_path, image)
                dest_path = os.path.join(animal_dest_path, image)
                shutil.copy2(source_path, dest_path)


def create_small_subsets():
    print("Creating small subsets of train, val, and test sets")
    copy_subset_of_images(train_dir, train_dir_small, 0.3)
    copy_subset_of_images(val_dir, val_dir_small, 0.1)
    copy_subset_of_images(test_dir, test_dir_small, 0.1)


def check_images_sizes():
    image_sizes = {}
    dirs = [train_dir, val_dir, test_dir]
    for dir in dirs:
        print(dir)
        animals = os.listdir(dir)
        for animal in animals:
            images = os.listdir(os.path.join(dir, animal))
            for image in images:
                image_path = os.path.join(dir, animal, image)
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_size = (width, height)
                    if image_size not in image_sizes:
                        image_sizes[image_size] = 1
                    else:
                        image_sizes[image_size] += 1
    return image_sizes


def plot_images_sizes(images_sizes):
    sorted_sizes = dict(
        sorted(images_sizes.items(), key=lambda item: item[1], reverse=True)
    )
    top_sizes = dict(list(sorted_sizes.items())[:10])
    plt.figure(figsize=(10, 6))
    plt.pie(top_sizes.values(), labels=top_sizes.keys(), autopct="%1.1f%%")
    plt.title("Top 10 Image Sizes")
    plt.tight_layout()
    plt.show()


def plot_images_grid(animals_folder):
    for animal_folder in animals_folder:
        animal_path = os.path.join(IMAGES_DATASET, animal_folder)
        if os.path.isdir(animal_path):
            images = os.listdir(animal_path)
            random_images = random.sample(images, 18)
            fig, ax = plt.subplots(3, 6, figsize=(15, 10))
            fig.suptitle(animal_folder, fontsize=16)
            for i, image in enumerate(random_images):
                image_path = os.path.join(animal_path, image)
                with Image.open(image_path) as img:
                    row = i // 6
                    col = i % 6
                    ax[row, col].imshow(img)
                    ax[row, col].axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    count_images()
    plot_images_grid(os.listdir(IMAGES_DATASET))
    find_invalid_images()
    image_counts = count_images_per_animal()
    plot_histo(image_counts)
    split_dataset(IMAGES_DATASET, train_dir, val_dir, test_dir)
    check_split()
    plot_images_sizes(check_images_sizes())
    create_small_subsets()
    print("Done")
