"""
SLM-Lab: LoRA Adapter Merging Script
===================================

This script merges LoRA adapters back into the base model to create a single,
standalone model that can be used for inference without requiring the original
base model or PEFT library.

What this script does:
1. Loads the original base model
2. Loads the trained LoRA adapters
3. Merges the adapter weights into the base model
4. Saves the merged model as a standalone checkpoint

Why merge adapters?
- Simpler deployment (no need for PEFT library)
- Faster inference (no adapter overhead)
- Easier model sharing and distribution
- Required for ONNX export and some deployment scenarios

Usage:
    python merge_adapters.py

Environment Variables (optional):
    BASE_MODEL_ID: Original base model (default: TinyLlama-1.1B)
    ADAPTER_DIR: Directory with LoRA adapters (default: ./outputs/tinyllama-lora)
    MERGED_OUT: Output directory for merged model (default: ./outputs/tinyllama-merged)
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# These environment variables allow you to customize the merging process

# Original base model that was fine-tuned
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Directory containing the trained LoRA adapters (output from train_lora.py)
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./outputs/tinyllama-lora")

# Output directory for the merged model
MERGED_OUT = os.environ.get("MERGED_OUT", "./outputs/tinyllama-merged")

print(f"Merging Configuration:")
print(f"  Base Model: {BASE_MODEL_ID}")
print(f"  Adapter Directory: {ADAPTER_DIR}")
print(f"  Output Directory: {MERGED_OUT}")

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================
# Load the tokenizer and base model

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

print("Loading base model...")
# Load the original base model in bfloat16 for memory efficiency
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    device_map="auto"            # Automatically place on available GPU/CPU
)

# ============================================================================
# ADAPTER MERGING SECTION
# ============================================================================
# Load and merge the LoRA adapters into the base model

print("Loading LoRA adapters...")
# Load the trained LoRA adapters
model = PeftModel.from_pretrained(base, ADAPTER_DIR)

print("Merging adapters into base model...")
# Merge the LoRA weights into the base model
# This creates a single model that combines the original weights with the fine-tuned changes
model = model.merge_and_unload()

# ============================================================================
# SAVE MERGED MODEL SECTION
# ============================================================================
# Save the merged model and tokenizer

print("Creating output directory...")
os.makedirs(MERGED_OUT, exist_ok=True)

print("Saving merged model...")
# Save the merged model with safe serialization (recommended for large models)
model.save_pretrained(MERGED_OUT, safe_serialization=True)

print("Saving tokenizer...")
# Save the tokenizer alongside the model
tok.save_pretrained(MERGED_OUT)

print(f"✅ Merging complete! Model saved to: {MERGED_OUT}")
print(f"📁 Next steps:")
print(f"   1. Test inference: python quantized_infer.py")
print(f"   2. Export to ONNX: python export_onnx.py")
print(f"   3. Serve with FastAPI: uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000")
print(f"")
print(f"💡 The merged model can now be used without the PEFT library!")
