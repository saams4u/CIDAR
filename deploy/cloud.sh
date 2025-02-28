#!/bin/bash

# cloud.sh for HubStack AI CIDAR Challenge Solution
# -------------------------------------------------
# This script deploys and runs the inference pipeline on cloud platforms (e.g., AWS EC2, GCP).
# It utilizes the optimized model to leverage cloud GPUs for high-speed inference.

# Exit on any error
set -e

# --------------------------- Configuration --------------------------- #
MODEL_NAME=${1:-"ViT"}
CHECKPOINT_PATH="../models/optimized/${MODEL_NAME}_tensorrt.onnx"
DATA_DIR="../data/processed/test"
OUTPUT_CSV="inference_results_cloud.csv"
INSTANCE_TYPE=${2:-"g4dn.xlarge"}  # Default AWS instance type with GPU support

# --------------------------- Helper Functions --------------------------- #
usage() {
  echo "Usage: $0 [MODEL] [INSTANCE_TYPE (optional)]"
  echo "Supported models: ViT, ConvNeXtV3, TFT"
  echo "Example:"
  echo "  $0 ViT g5.xlarge"
  exit 1
}

# Check if model is valid
if [[ ! " ViT ConvNeXtV3 TFT " =~ " ${MODEL_NAME} " ]]; then
  echo "[ERROR] Invalid model name: ${MODEL_NAME}"
  usage
fi

# Check if the optimized model exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "[ERROR] Optimized model not found at: $CHECKPOINT_PATH"
  echo "Please run the optimization step before deploying to cloud."
  exit 1
fi

# --------------------------- Run Inference --------------------------- #
echo "🚀 Running cloud inference with model: ${MODEL_NAME} on instance type: ${INSTANCE_TYPE}"

python src/test.py \
  --model "$MODEL_NAME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --data_dir "$DATA_DIR" \
  --output_csv "$OUTPUT_CSV"

echo "✅ Cloud inference completed. Results saved to $OUTPUT_CSV"