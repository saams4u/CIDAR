# Dockerfile for HubStack AI CIDAR Challenge Solution

# --------------------------- Base Image --------------------------- #
# Use NVIDIA PyTorch image with CUDA support for GPU acceleration
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# --------------------------- System Setup --------------------------- #
# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

# --------------------------- Python Dependencies --------------------------- #
# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------- Project Files --------------------------- #
# Copy the source code, models, and data directories into the container
COPY ./src ./src
COPY ./models ./models
COPY ./data ./data
COPY ./deploy ./deploy
COPY ./README.md ./README.md

# --------------------------- Environment Variables --------------------------- #
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# --------------------------- Entrypoint --------------------------- #
# Set the default command to bash for manual interaction
CMD ["/bin/bash"]

# --------------------------- Instructions --------------------------- #
# To build the Docker image:
# docker build -t hubstack-cidar-solution .
#
# To run the container with GPU support:
# docker run --gpus all -it --rm \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     hubstack-cidar-solution
#
# To run training inside the container:
# docker run --gpus all -it --rm \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     hubstack-cidar-solution \
#     python src/train.py --model ViT
#
# To run inference inside the container:
# docker run --gpus all -it --rm \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     hubstack-cidar-solution \
#     python src/test.py --model ViT --checkpoint models/checkpoints/ViT_best_model.pth