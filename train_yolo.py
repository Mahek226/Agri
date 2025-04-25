import os
import shutil
import argparse
import json
import yaml
from datetime import datetime
from utils import DatasetManager, ModelTrainer, ModelManager

BASE_DIR = "D:\MAHEK\Agriculture\datasets"

def main(args):
    model_type = args.model_type
    dataset_type = args.dataset_type
    selected_model = args.model
    selected_dataset_versions = args.versions
    run_name = args.run_name or f"{selected_model}_{datetime.today().strftime('%Y-%m-%d')}_v"
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    model_size = args.model_size

    DATASET_DIR = os.path.join(BASE_DIR, dataset_type.lower())
    temp_dataset = os.path.join(BASE_DIR, "temp_dataset")
    os.makedirs(temp_dataset, exist_ok=True)

    try:
        if dataset_type == "Classification":
            for version in selected_dataset_versions:
                DatasetManager.copy_classification_dataset(
                    os.path.join(DATASET_DIR, selected_model, version),
                    temp_dataset
                )
            trainer = ModelTrainer(run_name, model_size, dataset_type)
            trainer.train(temp_dataset, epochs, learning_rate)
            model_path = os.path.join("runs", "classify", run_name)

        elif dataset_type == "Detection":
            notes_file = os.path.join(DATASET_DIR, selected_model, selected_dataset_versions[0], "notes.json")
            with open(notes_file) as f:
                notes_data = json.load(f)
                labels = [cat["name"] for cat in notes_data["categories"]]

            DatasetManager.copy_detection_dataset(
                selected_dataset_versions,
                os.path.join(DATASET_DIR, selected_model),
                temp_dataset
            )

            yaml_path = os.path.join(temp_dataset, "data.yaml")
            with open(yaml_path, "w") as yf:
                yaml.dump({
                    "path": os.getcwd(),
                    "train": os.path.join(temp_dataset, "train", "images"),
                    "val": os.path.join(temp_dataset, "valid", "images"),
                    "test": os.path.join(temp_dataset, "test", "images"),
                    "names": labels,
                    "nc": len(labels)
                }, yf)

            trainer = ModelTrainer(run_name, model_size, dataset_type)
            trainer.train(yaml_path, epochs, learning_rate)
            model_path = os.path.join("runs", "detect", run_name)

        dest_path = os.path.join("runs", dataset_type, selected_model, run_name)
        ModelManager.copy_model(model_path, dest_path, run_name)
        print(f"[SUCCESS] Training completed for: {run_name}")

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
    finally:
        shutil.rmtree(temp_dataset)
        print("[INFO] Temporary files cleaned up.")

if __name__ == "_main_":
    parser = argparse.ArgumentParser(description="YOLOv8 Model Training Script")
    parser.add_argument("--model_type", type=str, default="Yolo")
    parser.add_argument("--dataset_type", type=str, choices=["Classification", "Detection"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--versions", nargs="+", required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model_size", type=str, choices=["n", "s", "m", "l", "x"], default="n")
    args = parser.parse_args()

    main(args)