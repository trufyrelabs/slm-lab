"""
SLM-Lab: Quantized Inference Testing Script
==========================================

This script loads a trained model and runs a quick inference test to verify
that training was successful and the model can generate reasonable responses.

What this script does:
1. Loads the trained model (merged or adapter format)
2. Applies 8-bit quantization for memory efficiency
3. Runs a test prompt to generate text
4. Displays the generated response

Key Concepts:
- 8-bit Quantization: Reduces memory usage by ~50% with minimal quality loss
- Inference: Using the trained model to generate text
- Memory Efficiency: Important for deployment on limited hardware

Usage:
    python quantized_infer.py

Notes:
- This script expects a merged model by default
- For adapter models, change MODEL_DIR to your adapter directory
- The test prompt can be customized for your specific use case
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# Model directory - change this if using adapters instead of merged model

# Default: Use merged model (output from merge_adapters.py)
MODEL_DIR = "./outputs/tinyllama-merged"

# Alternative: Use adapter model (output from train_lora.py)
# MODEL_DIR = "./outputs/tinyllama-lora"  # Uncomment to use adapters

# Test prompt - customize this for your specific use case
TEST_PROMPT = "Explain the difference between SLMs and LLMs in one paragraph."

print(f"Inference Configuration:")
print(f"  Model Directory: {MODEL_DIR}")
print(f"  Test Prompt: {TEST_PROMPT}")

# ============================================================================
# QUANTIZATION CONFIGURATION SECTION
# ============================================================================
# Set up 8-bit quantization for memory-efficient inference

print("Setting up 8-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Load model in 8-bit precision
)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================
# Load the tokenizer and model with quantization

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

print("Loading model with 8-bit quantization...")
# Load the model with 8-bit quantization for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    device_map="auto",              # Automatically place on GPU/CPU
    quantization_config=bnb_config  # Apply 8-bit quantization
)

print(f"Model loaded successfully!")
print(f"Model device: {model.device}")

# ============================================================================
# INFERENCE SECTION
# ============================================================================
# Run the test prompt and generate a response

print(f"\nGenerating response for: '{TEST_PROMPT}'")
print("-" * 50)

# Tokenize the input prompt
inputs = tok(TEST_PROMPT, return_tensors="pt").to(model.device)

# Generate text
# max_new_tokens: Maximum number of new tokens to generate
# do_sample: Use sampling instead of greedy decoding
# temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
# top_p: Nucleus sampling parameter
# pad_token_id: Token to use for padding
out = model.generate(
    **inputs, 
    max_new_tokens=200,      # Generate up to 200 new tokens
    do_sample=True,          # Use sampling for more diverse outputs
    temperature=0.7,         # Moderate randomness
    top_p=0.9,              # Nucleus sampling
    pad_token_id=tok.eos_token_id  # Use EOS token for padding
)

# Decode and display the generated text
generated_text = tok.decode(out[0], skip_special_tokens=True)
print(generated_text)
print("-" * 50)

print(f"✅ Inference test complete!")
print(f"💡 You can modify TEST_PROMPT to test different scenarios")
print(f"💡 Adjust generation parameters (temperature, max_new_tokens) as needed")

# ============================================================================
# CUSTOMIZATION TIPS
# ============================================================================
"""
To customize this script for your use case:

1. Change the test prompt:
   TEST_PROMPT = "Your custom prompt here"

2. Adjust generation parameters:
   - max_new_tokens: Longer/shorter responses
   - temperature: More/less random outputs
   - top_p: Different sampling strategy

3. Use different model formats:
   - Merged model: MODEL_DIR = "./outputs/tinyllama-merged"
   - Adapter model: MODEL_DIR = "./outputs/tinyllama-lora"

4. Disable quantization (if you have enough memory):
   - Remove bnb_config and quantization_config
   - Use regular model loading

5. Add more test cases:
   - Create a list of test prompts
   - Loop through them for comprehensive testing
"""
