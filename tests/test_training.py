
import os
import unittest

import torch
import numpy as np

from torch.utils.data import DataLoader
from src.train import get_model, train, validate, MultiSpectralDataset
from torchvision import transforms

# --------------------------- Configuration --------------------------- #
DATA_DIR = "../data/processed/train"
VAL_DIR = "../data/processed/val"
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(VAL_DIR, exist_ok=True)

        # Create sample training data
        for i in range(5):
            sample_image = np.random.rand(3, *IMAGE_SIZE).astype(np.float32)
            sample_label = float(i)  # Example labels
            np.save(os.path.join(DATA_DIR, f"sample_{i}_{sample_label}.npy"), sample_image)

        # Create sample validation data
        for i in range(3):
            sample_image = np.random.rand(3, *IMAGE_SIZE).astype(np.float32)
            sample_label = float(i)  # Example labels
            np.save(os.path.join(VAL_DIR, f"val_sample_{i}_{sample_label}.npy"), sample_image)

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def tearDown(self):
        # Clean up created sample data
        for folder in [DATA_DIR, VAL_DIR]:
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))

    def test_model_initialization(self):
        model = get_model("ViT").to(DEVICE)
        self.assertIsNotNone(model)
        self.assertTrue(any(param.requires_grad for param in model.parameters()))

    def test_training_step(self):
        model = get_model("ViT").to(DEVICE)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()

        train_dataset = MultiSpectralDataset(DATA_DIR, self.transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        train_loss = train(model, train_loader, criterion, optimizer, scaler)
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0.0)

    def test_validation_step(self):
        model = get_model("ViT").to(DEVICE)
        criterion = torch.nn.MSELoss()

        val_dataset = MultiSpectralDataset(VAL_DIR, self.transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        val_loss = validate(model, val_loader, criterion)
        self.assertIsInstance(val_loss, float)
        self.assertGreater(val_loss, 0.0)

if __name__ == "__main__":
    unittest.main()