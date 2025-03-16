import os
import shutil
import random

dataset_path = "pricetag_data"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

output_path = "datasets"

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, split, "labels"), exist_ok=True)

image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg")]
random.shuffle(image_files) 

train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1 

train_count = int(len(image_files) * train_ratio)
val_count = int(len(image_files) * val_ratio)
test_count = len(image_files) - train_count - val_count

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

def copy_files(file_list, split):
    for img_file in file_list:
        img_src = os.path.join(images_path, img_file)
        lbl_src = os.path.join(labels_path, img_file.replace(".jpg", ".txt"))

        img_dst = os.path.join(output_path, split, "images", img_file)
        lbl_dst = os.path.join(output_path, split, "labels", img_file.replace(".jpg", ".txt"))

        shutil.copy(img_src, img_dst)
        if os.path.exists(lbl_src): 
            shutil.copy(lbl_src, lbl_dst)

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Hoàn tất chia dữ liệu!")
