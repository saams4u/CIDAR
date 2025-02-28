#!/bin/bash

# edge.sh for HubStack AI CIDAR Challenge Solution
# -------------------------------------------------
# This script deploys the optimized model for inference on edge devices (e.g., NVIDIA Jetson Orin NX).
# It sets up the environment, runs inference using the TensorRT-optimized model, and logs results.

# Exit on any error
set -e

# --------------------------- Configuration --------------------------- #
MODEL_NAME=${1:-"ViT"}
CHECKPOINT_PATH="../models/optimized/${MODEL_NAME}_tensorrt.onnx"
DATA_DIR="../data/processed/test"
OUTPUT_CSV="inference_results_edge.csv"

# --------------------------- Helper Functions --------------------------- #
usage() {
  echo "Usage: $0 [MODEL]"
  echo "Supported models: ViT, ConvNeXtV3, TFT"
  echo "Example:"
  echo "  $0 ViT"
  exit 1
}

# Check if model is provided
if [[ ! " ViT ConvNeXtV3 TFT " =~ " ${MODEL_NAME} " ]]; then
  echo "[ERROR] Invalid model name: ${MODEL_NAME}"
  usage
fi

# Check if the optimized model exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "[ERROR] Optimized model not found at: $CHECKPOINT_PATH"
  echo "Please run the optimization step before deploying to edge."
  exit 1
fi

# --------------------------- Run Inference --------------------------- #
echo "🚀 Running edge inference with model: ${MODEL_NAME}" 

python src/test.py \
  --model "$MODEL_NAME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --data_dir "$DATA_DIR" \
  --output_csv "$OUTPUT_CSV"

echo "✅ Edge inference completed. Results saved to $OUTPUT_CSV"