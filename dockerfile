# SLM-Lab: GPU-Ready Docker Image for Model Serving
# ================================================
#
# This Dockerfile creates a containerized environment for serving fine-tuned
# Small Language Models using FastAPI. It's optimized for GPU inference
# and production deployment.
#
# What this Dockerfile does:
# 1. Uses NVIDIA CUDA base image for GPU support
# 2. Installs Python and required system dependencies
# 3. Installs Python packages for model serving
# 4. Sets up the FastAPI application
# 5. Exposes the API port for external access
#
# Key Features:
# - GPU support with CUDA 12.1.1
# - Optimized for inference performance
# - Production-ready configuration
# - Easy deployment and scaling
#
# Usage:
#   docker build -t slm-api .
#   docker run --gpus all -p 8000:8000 -e MODEL_DIR=/app/outputs/tinyllama-merged slm-api
#
# Environment Variables:
#   MODEL_DIR: Path to model directory (default: /app/outputs/tinyllama-merged)
#   CUDA_VISIBLE_DEVICES: GPU devices to use (optional)

# Use NVIDIA CUDA base image for GPU support
# This provides CUDA runtime and cuDNN libraries needed for GPU inference
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - python3-pip: Python package manager
# - git: For cloning repositories (if needed)
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for the application
WORKDIR /app

# Copy application files to the container
# This includes all Python scripts, requirements, and configuration files
COPY . /app

# Install Python dependencies
# Upgrade pip first, then install required packages for model serving
RUN pip3 install -U pip && \
    pip3 install -U \
    transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    fastapi \
    uvicorn[speed] \
    einops \
    safetensors

# Set default model directory
# This can be overridden when running the container
ENV MODEL_DIR=/app/outputs/tinyllama-merged

# Expose port 8000 for the FastAPI server
# This allows external access to the API endpoints
EXPOSE 8000

# Start the FastAPI server when the container runs
# uvicorn is the ASGI server that runs the FastAPI application
CMD ["uvicorn", "serve_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# USAGE EXAMPLES AND CUSTOMIZATION
# ============================================================================
#
# Basic usage:
#   docker build -t slm-api .
#   docker run --gpus all -p 8000:8000 slm-api
#
# With custom model directory:
#   docker run --gpus all -p 8000:8000 \
#     -v $PWD/outputs/tinyllama-merged:/models/merged \
#     -e MODEL_DIR=/models/merged slm-api
#
# With specific GPU:
#   docker run --gpus '"device=0"' -p 8000:8000 slm-api
#
# With resource limits:
#   docker run --gpus all -p 8000:8000 \
#     --memory=8g --cpus=4 slm-api
#
# For production deployment:
#   docker run --gpus all -p 8000:8000 \
#     --restart unless-stopped \
#     --name slm-api \
#     -v /path/to/models:/models \
#     -e MODEL_DIR=/models/your-model \
#     slm-api
#
# Health check:
#   curl http://localhost:8000/health
#
# API documentation:
#   Open http://localhost:8000/docs in your browser
#
# ============================================================================
# TROUBLESHOOTING
# ============================================================================
#
# If you get CUDA errors:
# - Ensure nvidia-docker is installed
# - Check that your GPU supports CUDA 12.1
# - Verify GPU drivers are up to date
#
# If the model doesn't load:
# - Check that MODEL_DIR points to a valid model
# - Ensure the model was merged (not just adapters)
# - Verify sufficient GPU memory
#
# If the server doesn't start:
# - Check that port 8000 is available
# - Verify all dependencies are installed
# - Check container logs: docker logs <container_id>
#
# ============================================================================
# OPTIMIZATION TIPS
# ============================================================================
#
# For better performance:
# - Use multiple GPUs with --gpus all
# - Mount model directory as volume for faster loading
# - Set appropriate memory limits
# - Use production ASGI server like gunicorn
#
# For smaller image size:
# - Use multi-stage builds
# - Remove unnecessary packages
# - Use alpine base image (if CUDA not needed)
#
# For security:
# - Run as non-root user
# - Use specific package versions
# - Scan for vulnerabilities
# - Implement authentication
