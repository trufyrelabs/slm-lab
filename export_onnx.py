"""
SLM-Lab: ONNX Export and Quantization Script
===========================================

This script exports a trained model to ONNX format and applies INT8 quantization
for efficient deployment on CPU, edge devices, or cloud platforms.

What this script does:
1. Loads the trained model (must be merged, not adapters)
2. Exports the model to ONNX format
3. Applies dynamic INT8 quantization to reduce model size
4. Saves both FP32 and INT8 versions of the model

Key Concepts:
- ONNX: Open Neural Network Exchange format for cross-platform deployment
- INT8 Quantization: Reduces model size by ~75% with minimal quality loss
- Dynamic Quantization: Quantizes weights at runtime for optimal performance
- Cross-platform: ONNX models can run on CPU, GPU, mobile, and edge devices

Why export to ONNX?
- Faster inference on CPU compared to PyTorch
- Smaller model size for deployment
- Cross-platform compatibility
- Integration with production systems
- Edge device deployment

Usage:
    python export_onnx.py

Prerequisites:
- Must use a merged model (run merge_adapters.py first)
- Requires optimum and onnxruntime packages
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.exporters.onnx import export, TasksManager
from onnxruntime.quantization import quantize_dynamic, QuantType

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# Model and output directories

# Input: Merged model directory (output from merge_adapters.py)
MODEL_DIR = "./outputs/tinyllama-merged"

# Output: Directory for ONNX models
ONNX_DIR = "./onnx_export"

print(f"ONNX Export Configuration:")
print(f"  Input Model: {MODEL_DIR}")
print(f"  Output Directory: {ONNX_DIR}")

# Create output directory
os.makedirs(ONNX_DIR, exist_ok=True)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================
# Load the tokenizer and model for export

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

print("Loading model for ONNX export...")
# Load the model in float16 for export (ONNX export works better with float16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.float16,  # Use float16 for export
    device_map="cpu"            # Export on CPU for compatibility
)

# ============================================================================
# ONNX EXPORT SECTION
# ============================================================================
# Export the model to ONNX format

print("Exporting model to ONNX format...")
print("This may take a few minutes depending on model size...")

# Export the model using Optimum
# This converts the PyTorch model to ONNX format
export(
    model=model,                    # The model to export
    config=model.config,            # Model configuration
    output=ONNX_DIR,               # Output directory
    task="text-generation",         # Task type for text generation
    opset=17,                      # ONNX opset version (latest stable)
    tokenizer=tok,                 # Tokenizer for preprocessing
)

print("ONNX export completed successfully!")

# ============================================================================
# QUANTIZATION SECTION
# ============================================================================
# Apply INT8 dynamic quantization to reduce model size

print("Applying INT8 dynamic quantization...")

# File paths for the models
model_fp32 = os.path.join(ONNX_DIR, "model.onnx")      # Original FP32 model
model_int8 = os.path.join(ONNX_DIR, "model-int8.onnx")  # Quantized INT8 model

# Apply dynamic quantization
# This reduces model size by ~75% with minimal quality loss
quantize_dynamic(
    model_input=model_fp32,        # Input FP32 model
    model_output=model_int8,       # Output INT8 model
    weight_type=QuantType.QInt8,   # Use INT8 quantization
)

print("INT8 quantization completed!")

# ============================================================================
# VERIFICATION SECTION
# ============================================================================
# Check the exported models

print("\nExport Summary:")
print("-" * 40)

# Check if files exist and get sizes
if os.path.exists(model_fp32):
    fp32_size = os.path.getsize(model_fp32) / (1024 * 1024)  # MB
    print(f"✅ FP32 Model: {model_fp32}")
    print(f"   Size: {fp32_size:.1f} MB")

if os.path.exists(model_int8):
    int8_size = os.path.getsize(model_int8) / (1024 * 1024)  # MB
    reduction = (1 - int8_size / fp32_size) * 100
    print(f"✅ INT8 Model: {model_int8}")
    print(f"   Size: {int8_size:.1f} MB ({reduction:.1f}% smaller)")

print(f"\n✅ ONNX export complete! Models saved to: {ONNX_DIR}")
print(f"📁 Next steps:")
print(f"   1. Use model-int8.onnx for CPU deployment")
print(f"   2. Use model.onnx for GPU deployment")
print(f"   3. Integrate with ONNX Runtime for inference")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
Example usage of the exported ONNX models:

1. Using ONNX Runtime (Python):
   ```python
   import onnxruntime as ort
   
   # Load the quantized model
   session = ort.InferenceSession("onnx_export/model-int8.onnx")
   
   # Run inference
   inputs = {"input_ids": input_ids}
   outputs = session.run(None, inputs)
   ```

2. Using ONNX Runtime (C++):
   ```cpp
   #include <onnxruntime_cxx_api.h>
   
   Ort::Session session(env, "model-int8.onnx", session_options);
   ```

3. Using ONNX Runtime Mobile (Android/iOS):
   ```java
   // Android example
   OrtSession session = env.createSession("model-int8.onnx");
   ```

4. Using TensorRT (NVIDIA GPUs):
   ```python
   import tensorrt as trt
   
   # Convert ONNX to TensorRT
   engine = trt.engine_from_network(onnx_model)
   ```

Benefits of the exported models:
- 75% smaller file size (INT8 vs FP32)
- Faster CPU inference
- Cross-platform compatibility
- Production-ready deployment
- Edge device support
"""
