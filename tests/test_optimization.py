
import os
import unittest

import torch

import numpy as np

import onnx
import onnxruntime as ort

from src.optimize import export_to_onnx, convert_to_torchscript, optimize_with_tensorrt
from src.train import get_model

# --------------------------- Configuration --------------------------- #
MODEL_NAME = "ViT"
CHECKPOINT_DIR = "../models/checkpoints/"
OPTIMIZED_DIR = "../models/optimized/"
IMAGE_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ONNX_OUTPUT_PATH = os.path.join(OPTIMIZED_DIR, f"{MODEL_NAME}.onnx")
TORCHSCRIPT_OUTPUT_PATH = os.path.join(OPTIMIZED_DIR, f"{MODEL_NAME}_torchscript.pt")
TENSORRT_OUTPUT_PATH = os.path.join(OPTIMIZED_DIR, f"{MODEL_NAME}_tensorrt.onnx")

class TestOptimizationPipeline(unittest.TestCase):

    def setUp(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(OPTIMIZED_DIR, exist_ok=True)

        # Create a dummy model checkpoint if not exists
        self.model = get_model(MODEL_NAME).to(DEVICE)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best_model.pth")
        if not os.path.exists(checkpoint_path):
            torch.save(self.model.state_dict(), checkpoint_path)

    def tearDown(self):
        # Clean up generated files
        for path in [ONNX_OUTPUT_PATH, TORCHSCRIPT_OUTPUT_PATH, TENSORRT_OUTPUT_PATH]:
            if os.path.exists(path):
                os.remove(path)

    def test_export_to_onnx(self):
        export_to_onnx(self.model, ONNX_OUTPUT_PATH, input_size=IMAGE_SIZE)
        self.assertTrue(os.path.exists(ONNX_OUTPUT_PATH))
        onnx_model = onnx.load(ONNX_OUTPUT_PATH)
        onnx.checker.check_model(onnx_model)

    def test_convert_to_torchscript(self):
        convert_to_torchscript(self.model, TORCHSCRIPT_OUTPUT_PATH)
        self.assertTrue(os.path.exists(TORCHSCRIPT_OUTPUT_PATH))
        loaded_script_model = torch.jit.load(TORCHSCRIPT_OUTPUT_PATH)
        self.assertIsInstance(loaded_script_model, torch.jit.ScriptModule)

    def test_optimize_with_tensorrt(self):
        export_to_onnx(self.model, ONNX_OUTPUT_PATH, input_size=IMAGE_SIZE)  # Ensure ONNX model exists
        optimize_with_tensorrt(ONNX_OUTPUT_PATH, TENSORRT_OUTPUT_PATH)
        self.assertTrue(os.path.exists(TENSORRT_OUTPUT_PATH))

        # Load and validate TensorRT-optimized ONNX model
        ort_session = ort.InferenceSession(TENSORRT_OUTPUT_PATH, providers=["TensorrtExecutionProvider"])
        dummy_input = np.random.randn(1, 3, *IMAGE_SIZE).astype(np.float32)
        outputs = ort_session.run(None, {"input": dummy_input})
        self.assertIsNotNone(outputs)
        self.assertIsInstance(outputs, list)
        self.assertGreater(len(outputs), 0)

if __name__ == "__main__":
    unittest.main()