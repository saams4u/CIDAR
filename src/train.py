"""
Model Training Script for HubStack AI CIDAR Challenge Solution
--------------------------------------------------------------
This script handles:
- Model definition and loading
- Training and validation loops
- Checkpoint saving and early stopping
- Mixed-precision training for faster convergence
- Hardware-aware optimizations

Author: HubStack AI, Inc.

## Features:
- **Supports multiple architectures:** Train ViT, ConvNeXtV3, and Temporal Fusion Transformer (TFT) models.
- **Mixed-precision training:** Leverages `torch.cuda.amp` for faster training and reduced memory usage.
- **Early stopping:** Prevents overfitting with customizable patience to halt training when validation loss stops improving.
- **Checkpointing:** Saves the best model based on validation loss, allowing for recovery and future inference.
- **Command-line arguments:** Easily select model architecture for flexible experimentation.
- **Batch processing:** Efficient data handling with PyTorch’s DataLoader for training and validation datasets.
- **Real-time progress tracking:** Displays progress bars for training and validation loops using `tqdm`.

## Usage Example:
# Train using the Vision Transformer (ViT) model:
python train.py --model ViT

# Train using the ConvNeXtV3 model:
python train.py --model ConvNeXtV3

# Train using the Temporal Fusion Transformer (TFT) model:
python train.py --model TFT

## Output Example:
# 🚀 Starting training with ViT...
#
# 🔄 Epoch 1/50
# Training: 100%|██████████| 200/200 [00:30<00:00,  6.58it/s]
# Validation: 100%|██████████| 50/50 [00:05<00:00,  9.45it/s]
# ✅ Epoch 1 | Train Loss: 0.0123 | Val Loss: 0.0098
#
# 🔄 Epoch 2/50
# Training: 100%|██████████| 200/200 [00:29<00:00,  6.75it/s]
# Validation: 100%|██████████| 50/50 [00:05<00:00,  9.50it/s]
# ✅ Epoch 2 | Train Loss: 0.0091 | Val Loss: 0.0085
#
# 🔔 Early stopping triggered!
# 🎯 Training complete!
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tqdm import tqdm
from typing import Tuple, Dict

from models import ViTModel, ConvNeXtV3, TemporalFusionTransformer
from utils import save_checkpoint, load_checkpoint, EarlyStopping

# ----------------------------- Configuration ----------------------------- #
DATA_DIR = "../data/processed/"
CHECKPOINT_DIR = "../checkpoints/"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)


# ----------------------------- Dataset Class ----------------------------- #
class MultiSpectralDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = list(sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.npy')]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = torch.tensor(np.load(self.image_paths[idx]), dtype=torch.float32)
        label = torch.tensor(float(self.image_paths[idx].split('_')[-1].replace('.npy', '')))  # Example label extraction
        if self.transform:
            image = self.transform(image)
        return image, label


# ----------------------------- Model Selection ----------------------------- #
def get_model(model_name: str) -> nn.Module:
    """Load the specified model architecture."""
    if model_name == 'ViT':
        return ViTModel(input_size=IMAGE_SIZE, num_classes=1)
    elif model_name == 'ConvNeXtV3':
        return ConvNeXtV3(input_size=IMAGE_SIZE, num_classes=1)
    elif model_name == 'TFT':
        return TemporalFusionTransformer(input_size=IMAGE_SIZE, num_classes=1)
    else:
        raise ValueError(f"[ERROR] Model {model_name} not supported.")


# ----------------------------- Training Loop ----------------------------- #
def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scaler: torch.cuda.amp.GradScaler) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(dataloader)


# ----------------------------- Validation Loop ----------------------------- #
def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

    return val_loss / len(dataloader)


# ----------------------------- Main Training Pipeline ----------------------------- #
def main(model_name: str = 'ViT'):
    print(f"🚀 Starting training with {model_name}...")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize model, criterion, optimizer, and scaler
    model = get_model(model_name).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Data loaders
    transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
    train_dataset = MultiSpectralDataset(os.path.join(DATA_DIR, 'train'), transform)
    val_dataset = MultiSpectralDataset(os.path.join(DATA_DIR, 'val'), transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    early_stopping = EarlyStopping(patience=5, verbose=True, path=os.path.join(CHECKPOINT_DIR, f"{model_name}_best_model.pth"))

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\n🔄 Epoch {epoch}/{EPOCHS}")
        train_loss = train(model, train_loader, criterion, optimizer, scaler)
        val_loss = validate(model, val_loader, criterion)

        print(f"✅ Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping and checkpointing
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("🔔 Early stopping triggered!")
            break

    print("🎯 Training complete!")


# ----------------------------- Entry Point ----------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train models for the HubStack AI CIDAR Challenge.")
    parser.add_argument("--model", type=str, default="ViT", choices=["ViT", "ConvNeXtV3", "TFT"], help="Model architecture to train.")
    args = parser.parse_args()

    main(model_name=args.model)