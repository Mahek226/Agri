import argparse
import os
import shutil
from utils import UploadFileManager
from utils import ClassificationDatasetSplitter#, DetectionDatasetSplitter
#from utils import BenchmarkProcessor

BASE_DIR = "datasets"
CLASS_DIR = os.path.join(BASE_DIR, "classification")
DETECT_DIR = os.path.join(BASE_DIR, "detection")
#BENCHMARK_DIR = "Benchmark"
#RUNS_DIR = "runs"

os.makedirs(CLASS_DIR, exist_ok=True)
os.makedirs(DETECT_DIR, exist_ok=True)

def upload_and_split(args):
    dataset_type = args.type
    model = args.model
    version = args.version
    zip_path = args.zip
    train = args.train
    val = args.val
    test = args.test

    if abs(train + val + test - 1.0) > 1e-6:
        print("❌ Split ratios must sum to 1.")
        return

    target_dir = CLASS_DIR if dataset_type == "classification" else DETECT_DIR
    path = os.path.join(target_dir, model, version)
    os.makedirs(path, exist_ok=True)

    zip_dest = os.path.join(path, os.path.basename(zip_path))
    shutil.copy(zip_path, zip_dest)

    print(f"📦 Extracting zip file to {path} ...")
    UploadFileManager.extract_zip(zip_dest, path)
    os.remove(zip_dest)

    print("🗃 Contents of dataset_path after extraction:", os.listdir(path))
    if dataset_type == "classification":
        ClassificationDatasetSplitter.split(path, train, val, test)
    # else:
    #     DetectionDatasetSplitter.split(path, train, val, test)

    print(f"✅ Dataset '{version}' processed successfully at `{path}`")

# def benchmark_model(args):
#     folder = args.folder
#     model = args.model
#     version = args.version
#     zip_path = args.zip

#     bench_path = os.path.join(BENCHMARK_DIR, folder, model, version)
#     os.makedirs(bench_path, exist_ok=True)

#     print(f"📊 Preparing benchmark at {bench_path} ...")
#     BenchmarkProcessor.prepare_benchmark(zip_path, bench_path, UploadFileManager)
#     print(f"✅ Benchmark prepared at `{bench_path}`")


def main():
    parser = argparse.ArgumentParser(description="Dataset Uploader and Benchmark CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload Command
    upload_parser = subparsers.add_parser("upload", help="Upload and split dataset")
    upload_parser.add_argument("--type", choices=["classification", "detection"], required=True)
    upload_parser.add_argument("--model", required=True, help="Model name")
    upload_parser.add_argument("--version", required=True, help="Dataset version")
    upload_parser.add_argument("--zip", required=True, help="Path to dataset zip")
    upload_parser.add_argument("--train", type=float, default=0.7)
    upload_parser.add_argument("--val", type=float, default=0.2)
    upload_parser.add_argument("--test", type=float, default=0.1)

    # Benchmark Command
    # bench_parser = subparsers.add_parser("benchmark", help="Run benchmark for a model")
    # bench_parser.add_argument("--folder", required=True, help="Folder inside 'runs/'")
    # bench_parser.add_argument("--model", required=True, help="Model name")
    # bench_parser.add_argument("--version", required=True, help="Benchmark version name")
    # bench_parser.add_argument("--zip", required=True, help="Path to benchmark zip file")

    args = parser.parse_args()

    if args.command == "upload":
        upload_and_split(args)
    # elif args.command == "benchmark":
    #     benchmark_model(args)

if __name__ == "__main__":
    main()
