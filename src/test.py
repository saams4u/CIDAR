"""
Inference Pipeline for HubStack AI CIDAR Challenge Solution
-----------------------------------------------------------
This script handles:
- Model loading from checkpoints
- Real-time data inference on multi-spectral imagery
- Latency measurement for edge and cloud deployments
- Support for hardware acceleration (TensorRT, PyTorch with CUDA)

Author: HubStack AI, Inc.

## Features:
- **Model loading:** Easily load pretrained models with support for ViT, ConvNeXtV3, and Temporal Fusion Transformer (TFT).
- **Real-time inference:** Perform batch inference with latency measurements for each sample.
- **Batch processing:** Efficiently process large datasets using PyTorch's DataLoader with progress bars.
- **Hardware acceleration:** Leverage CUDA-enabled GPUs for faster inference or run on CPU if unavailable.
- **Results output:** Save inference predictions into a CSV file with filenames and prediction values.
- **User-friendly interface:** Command-line arguments enable flexible input configurations.

## Usage Example:
# Run inference using the Vision Transformer (ViT) model:
python test.py --model ViT --checkpoint ../checkpoints/ViT_best_model.pth --data_dir ../data/processed/test --output_csv predictions.csv

# Run inference using the ConvNeXtV3 model:
python test.py --model ConvNeXtV3 --checkpoint ../checkpoints/ConvNeXtV3_best_model.pth --data_dir ../data/processed/test --output_csv predictions_convnext.csv

# Run inference using the Temporal Fusion Transformer (TFT) model:
python test.py --model TFT --checkpoint ../checkpoints/TFT_best_model.pth --data_dir ../data/processed/test --output_csv predictions_tft.csv

## Output Example:
# 📂 Preparing data...
# 🚦 Starting inference...
# 🚀 Running Inference: 100%|██████████| 50/50 [00:04<00:00, 12.00it/s]
# ⚡ Average Inference Latency: 45.20 ms per sample
# ✅ Inference results saved to predictions.csv
# 🏁 Inference completed.
"""

import os

import time
import torch

import numpy as np

from glob import glob
from tqdm import tqdm
from typing import Tuple

from torch.utils.data import DataLoader, Dataset

from models import ViTModel, ConvNeXtV3, TemporalFusionTransformer
from utils import load_checkpoint

from albumentations import Normalize
from albumentations.pytorch import ToTensorV2

import cv2

# --------------------------- Configuration --------------------------- #
DATA_DIR = "../data/processed/"
CHECKPOINT_DIR = "../checkpoints/"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)
SUPPORTED_MODELS = {"ViT": ViTModel, "ConvNeXtV3": ConvNeXtV3, "TFT": TemporalFusionTransformer}


# --------------------------- Dataset Class --------------------------- #
class InferenceDataset(Dataset):
    def __init__(self, data_dir: str):
        self.image_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        self.transform = Normalize(mean=(0.5,), std=(0.5,))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = np.load(self.image_paths[idx])
        image = self.transform(image=image)['image']
        return image, os.path.basename(self.image_paths[idx])


# --------------------------- Inference Functions --------------------------- #
def load_model(model_name: str, checkpoint_path: str) -> torch.nn.Module:
    """Load model architecture and weights."""
    model_class = SUPPORTED_MODELS.get(model_name)
    if model_class is None:
        raise ValueError(f"[ERROR] Model '{model_name}' is not supported.")
    model = model_class(input_size=IMAGE_SIZE, num_classes=1).to(DEVICE)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model


def run_inference(model: torch.nn.Module, dataloader: DataLoader) -> Tuple[float, list]:
    """Run inference on the dataset and measure latency."""
    results = []
    total_latency = 0.0

    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="🚀 Running Inference"):
            images = images.to(DEVICE)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            batch_latency = (end_time - start_time) * 1000  # ms
            total_latency += batch_latency

            for filename, output in zip(filenames, outputs.squeeze().cpu().numpy()):
                results.append((filename, float(output)))

    avg_latency = total_latency / len(dataloader.dataset)
    return avg_latency, results


def save_inference_results(results: list, output_path: str):
    """Save inference results to a CSV file."""
    import pandas as pd
    df = pd.DataFrame(results, columns=["filename", "prediction"])
    df.to_csv(output_path, index=False)
    print(f"✅ Inference results saved to {output_path}")


# --------------------------- Main Execution --------------------------- #
def main(model_name: str, checkpoint: str, data_dir: str, output_csv: str):
    print(f"🔍 Loading model: {model_name}")
    model = load_model(model_name, checkpoint)

    print("📂 Preparing data...")
    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("🚦 Starting inference...")
    avg_latency, results = run_inference(model, dataloader)
    print(f"⚡ Average Inference Latency: {avg_latency:.2f} ms per sample")

    save_inference_results(results, output_csv)
    print("🏁 Inference completed.")


# --------------------------- Entry Point --------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference pipeline for HubStack AI CIDAR Challenge.")
    parser.add_argument("--model", type=str, default="ViT", choices=["ViT", "ConvNeXtV3", "TFT"], help="Model architecture to use.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="../data/processed/test", help="Directory containing preprocessed images.")
    parser.add_argument("--output_csv", type=str, default="test.csv", help="Output CSV file for predictions.")

    args = parser.parse_args()

    main(
        model_name=args.model,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_csv=args.output_csv
    )