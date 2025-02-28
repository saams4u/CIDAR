
import os
import unittest
import torch

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from src.test import load_model, InferenceDataset, run_inference, save_inference_results

# --------------------------- Configuration --------------------------- #
DATA_DIR = "../data/processed/test"
CHECKPOINT_DIR = "../models/checkpoints/"
BATCH_SIZE = 4
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "ViT"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best_model.pth")
OUTPUT_CSV = "test_inference_results.csv"

class TestInferencePipeline(unittest.TestCase):

    def setUp(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # Create sample test data
        for i in range(3):
            sample_image = np.random.rand(3, *IMAGE_SIZE).astype(np.float32)
            np.save(os.path.join(DATA_DIR, f"test_sample_{i}.npy"), sample_image)

        # Create a dummy checkpoint (if not already present)
        if not os.path.exists(CHECKPOINT_PATH):
            dummy_model = load_model(MODEL_NAME, checkpoint_path=None)
            torch.save(dummy_model.state_dict(), CHECKPOINT_PATH)

    def tearDown(self):
        # Clean up created sample data and output files
        for file in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, file))
        if os.path.exists(OUTPUT_CSV):
            os.remove(OUTPUT_CSV)

    def test_model_loading(self):
        model = load_model(MODEL_NAME, CHECKPOINT_PATH)
        self.assertIsNotNone(model)
        self.assertTrue(any(param.requires_grad for param in model.parameters()))

    def test_inference_execution(self):
        model = load_model(MODEL_NAME, CHECKPOINT_PATH)
        dataset = InferenceDataset(DATA_DIR)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        avg_latency, results = run_inference(model, dataloader)

        self.assertIsInstance(avg_latency, float)
        self.assertGreater(len(results), 0)
        for filename, prediction in results:
            self.assertIsInstance(filename, str)
            self.assertIsInstance(prediction, float)

    def test_inference_result_saving(self):
        model = load_model(MODEL_NAME, CHECKPOINT_PATH)
        dataset = InferenceDataset(DATA_DIR)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        _, results = run_inference(model, dataloader)
        save_inference_results(results, OUTPUT_CSV)

        self.assertTrue(os.path.exists(OUTPUT_CSV))
        df = pd.read_csv(OUTPUT_CSV)
        self.assertFalse(df.empty)
        self.assertIn("filename", df.columns)
        self.assertIn("prediction", df.columns)

if __name__ == "__main__":
    unittest.main()