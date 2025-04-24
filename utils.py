import os
import random
import shutil
import zipfile
import streamlit as st
from pathlib import Path


# ----------------- UTILITY CLASSES FOR UPLOAD_DATASET-----------------
class UploadFileManager:
    @staticmethod
    def extract_zip(zip_path, extract_to):
        import pprint
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            temp_extract_path = os.path.join(extract_to, "temp_extracted")
            zip_ref.extractall(temp_extract_path)

            moved_files_count = 0

            for root, dirs, files in os.walk(temp_extract_path):
                # Look for class folders that contain images (e.g., .jpg, .png)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    class_name = os.path.basename(root)
                    class_dir = os.path.join(extract_to, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    for file in image_files:
                        src = os.path.join(root, file)
                        dst = os.path.join(class_dir, file)
                        shutil.move(src, dst)
                        moved_files_count += 1

            print(f"✅ Total image files moved: {moved_files_count}")
            shutil.rmtree(temp_extract_path)



class ClassificationDatasetSplitter:
    @staticmethod
    def split(dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        categories = [cat for cat in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cat))]
        print("📁 Categories found:", categories)
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(dataset_path, split), exist_ok=True)

        for category in categories:
            category_path = os.path.join(dataset_path, category)
            images = [img for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img))]
            print(f"📷 Found {len(images)} images in '{category}'")
            random.shuffle(images)

            if not images: continue

            train_size = int(len(images) * train_ratio)
            val_size = int(len(images) * val_ratio)
            train, val, test = images[:train_size], images[train_size:train_size+val_size], images[train_size+val_size:]

            for split_name, files in zip(['train', 'valid', 'test'], [train, val, test]):
                split_dir = os.path.join(dataset_path, split_name, category)
                os.makedirs(split_dir, exist_ok=True)
                for img in files:
                    shutil.move(os.path.join(category_path, img), os.path.join(split_dir, img))

            if not os.listdir(category_path):
                os.rmdir(category_path)
