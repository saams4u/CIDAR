#!/bin/bash

# entrypoint.sh for HubStack AI CIDAR Challenge Solution
# ------------------------------------------------------
# This script serves as the container entrypoint, providing options to run preprocessing,
# training, inference, or optimization pipelines directly from the Docker container.

# Exit immediately if a command exits with a non-zero status
set -e

# Display usage instructions
usage() {
  echo "Usage: $0 [command] [options]"
  echo "Commands:"
  echo "  preprocess             Run data preprocessing pipeline"
  echo "  train [MODEL]          Train the specified model (ViT, ConvNeXtV3, TFT)"
  echo "  test [MODEL]           Run inference with the specified model"
  echo "  optimize [MODEL]       Optimize the specified model for deployment"
  echo "  bash                   Start an interactive bash session"
  echo ""
  echo "Examples:"
  echo "  $0 preprocess"
  echo "  $0 train ViT"
  echo "  $0 test ConvNeXtV3"
  echo "  $0 optimize TFT"
  echo "  $0 bash"
  exit 1
}

# Check for at least one argument
if [ "$#" -lt 1 ]; then
  usage
fi

COMMAND=$1
MODEL=$2

case $COMMAND in
  preprocess)
    echo "🚀 Running data preprocessing..."
    python src/preprocess.py
    ;;

  train)
    if [ -z "$MODEL" ]; then
      echo "[ERROR] Model name required. Supported models: ViT, ConvNeXtV3, TFT"
      usage
    fi
    echo "🚀 Training model: $MODEL..."
    python src/train.py --model "$MODEL"
    ;;

  test)
    if [ -z "$MODEL" ]; then
      echo "[ERROR] Model name required. Supported models: ViT, ConvNeXtV3, TFT"
      usage
    fi
    echo "🚀 Running inference with model: $MODEL..."
    python src/test.py --model "$MODEL" --checkpoint models/checkpoints/${MODEL}_best_model.pth
    ;;

  optimize)
    if [ -z "$MODEL" ]; then
      echo "[ERROR] Model name required. Supported models: ViT, ConvNeXtV3, TFT"
      usage
    fi
    echo "🚀 Optimizing model: $MODEL..."
    python src/optimize.py --model "$MODEL" --checkpoint models/checkpoints/${MODEL}_best_model.pth
    ;;

  bash)
    echo "🔔 Starting interactive bash session..."
    exec "/bin/bash"
    ;;

  *)
    echo "[ERROR] Unknown command: $COMMAND"
    usage
    ;;
esac