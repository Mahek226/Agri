import os
import random
import shutil
import zipfile
from pathlib import Path
from ultralytics import YOLO


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

# ----------------- UTILITY CLASSES FOR TRAINING_YOLO_MODEL-----------------
class DatasetManager:
    @staticmethod
    def copy_classification_dataset(source_path, dest_path):
        for phase in ["train", "valid", "test"]:
            src = os.path.join(source_path, phase)
            dest = os.path.join(dest_path, phase)
            if os.path.exists(src):
                for class_dir in os.listdir(src):
                    src_class = os.path.join(src, class_dir)
                    dest_class = os.path.join(dest, class_dir)
                    os.makedirs(dest_class, exist_ok=True)
                    for file in os.listdir(src_class):
                        shutil.copy2(os.path.join(src_class, file), os.path.join(dest_class, file))

    @staticmethod
    def copy_detection_dataset(selected_versions, src_base, dest_base):
        split_map = {"train": "train", "validate": "valid", "test": "test"}
        for original, target in split_map.items():
            for sub in ["images", "labels"]:
                for version in selected_versions:
                    src_path = os.path.join(src_base, version, original, sub)
                    dest_path = os.path.join(dest_base, target, sub)
                    if os.path.exists(src_path):
                        os.makedirs(dest_path, exist_ok=True)
                        for file in os.listdir(src_path):
                            shutil.copy2(os.path.join(src_path, file), os.path.join(dest_path, file))


class ModelTrainer:
    def _init_(self, run_name, model_size, dataset_type):
        self.run_name = run_name
        self.model_size = model_size
        self.dataset_type = dataset_type
        self.trainer = YOLO(f"yolov8{model_size}{'-cls' if dataset_type == 'Classification' else ''}.pt")

    def train(self, data_path, epochs, lr):
        self.trainer.train(
            data=data_path,
            model=self.trainer,
            epochs=epochs,
            lr0=lr,
            name=self.run_name
        )


class ModelManager:
    @staticmethod
    def copy_model(src_folder, dest_folder, run_name):
        shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
        weights_folder = os.path.join(dest_folder, "weights")
        best_model = os.path.join(weights_folder, "best.pt")
        renamed_model = os.path.join(weights_folder, f"{run_name}.pt")

        if os.path.exists(best_model):
            os.rename(best_model, renamed_model)
            print(f"[INFO] Model renamed and saved at: {renamed_model}")
        else:
            print("[ERROR] best.pt not found in weights folder.")