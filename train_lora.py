"""
SLM-Lab: LoRA/QLoRA Fine-tuning Script
=====================================

This script fine-tunes a Small Language Model using LoRA (Low-Rank Adaptation) or QLoRA
(Quantized LoRA) for efficient training with limited GPU memory.

What this script does:
1. Loads a base model (e.g., TinyLlama, Phi-2, Mistral)
2. Prepares training data in instruction format
3. Applies LoRA/QLoRA for parameter-efficient fine-tuning
4. Trains the model on your custom dataset
5. Saves the trained adapter weights

Key Concepts:
- LoRA: Adds small trainable matrices to existing layers (memory efficient)
- QLoRA: Uses 4-bit quantization during training (even more memory efficient)
- Instruction Tuning: Trains model to follow instructions with examples

Usage:
    python train_lora.py

Environment Variables (optional):
    BASE_MODEL_ID: Model to fine-tune (default: TinyLlama-1.1B)
    USE_QLORA: Use QLoRA (1) or LoRA (0) (default: 1)
    BATCH_SIZE: Training batch size (default: 2)
    GRAD_ACCUM: Gradient accumulation steps (default: 8)
    EPOCHS: Number of training epochs (default: 2)
    LR: Learning rate (default: 2e-4)
    MAX_SEQ_LEN: Maximum sequence length (default: 1024)
"""

from dataclasses import dataclass
from typing import Dict, List
import os

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# These environment variables allow you to customize training without editing code
# Set them before running: export BASE_MODEL_ID="microsoft/phi-2"

# Base model to fine-tune (choose from Hugging Face Hub)
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Output directory for saved model and checkpoints
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/tinyllama-lora")

# Training method: QLoRA (4-bit quantized) or LoRA (16-bit)
# QLoRA uses less memory but may be slightly slower
USE_QLORA = bool(int(os.environ.get("USE_QLORA", "1")))  # 1=QLoRA (4-bit), 0=LoRA (fp16/bf16)

# Training hyperparameters
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))  # Maximum tokens per example
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))       # Examples per batch
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))       # Accumulate gradients over N steps
LR = float(os.environ.get("LR", "2e-4"))                  # Learning rate
EPOCHS = float(os.environ.get("EPOCHS", "2"))             # Number of training epochs
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "50"))  # Learning rate warmup
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "500"))     # Save checkpoint every N steps
LOG_STEPS = int(os.environ.get("LOG_STEPS", "25"))        # Log metrics every N steps

print(f"Training Configuration:")
print(f"  Base Model: {BASE_MODEL_ID}")
print(f"  Method: {'QLoRA (4-bit)' if USE_QLORA else 'LoRA (16-bit)'}")
print(f"  Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
print(f"  Learning Rate: {LR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Max Sequence Length: {MAX_SEQ_LEN}")

# ============================================================================
# DATA PREPARATION SECTION
# ============================================================================
# Load your training data from JSONL file
# Expected format: {"instruction": "...", "input": "...", "output": "..."}

DATA_PATH = os.environ.get("DATA_PATH", "./data/instructions.jsonl")
print(f"Loading dataset from: {DATA_PATH}")

# Load dataset from JSONL file
DATASET = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Dataset loaded: {len(DATASET)} examples")

def format_example(ex):
    """
    Convert raw data into instruction format for training.
    
    This function takes a raw example and formats it into the instruction
    format that the model will learn to follow.
    
    Args:
        ex: Dictionary with 'instruction', 'input' (optional), 'output' fields
    
    Returns:
        Dictionary with 'text' field containing formatted prompt + response
    """
    instruction = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    
    # Format as instruction-following prompt
    if inp:
        # If input is provided, include it in the prompt
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        # If no input, just use instruction
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    return {"text": prompt + out}

# Apply formatting to all examples
DATASET = DATASET.map(format_example, remove_columns=DATASET.column_names)
print("Dataset formatted for instruction tuning")

# ============================================================================
# TOKENIZER & MODEL SETUP SECTION
# ============================================================================
# Load tokenizer and prepare it for training

print(f"Loading tokenizer from: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

# Set padding token if not present (required for training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

def tokenize_fn(batch):
    """
    Tokenize text data for training.
    
    This function converts text into token IDs that the model can process.
    It handles padding, truncation, and returns the proper format for training.
    
    Args:
        batch: Batch of text examples
    
    Returns:
        Tokenized batch with input_ids and attention_mask
    """
    return tokenizer(
        batch["text"],
        truncation=True,           # Cut sequences longer than max_length
        max_length=MAX_SEQ_LEN,    # Maximum sequence length
        padding="max_length",      # Pad shorter sequences to max_length
        return_tensors=None,       # Return Python lists, not tensors
    )

# Tokenize the entire dataset
print("Tokenizing dataset...")
tokenized = DATASET.map(tokenize_fn, batched=True, remove_columns=["text"])
print("Tokenization complete")

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================
# Load the base model with appropriate configuration

print(f"Loading base model: {BASE_MODEL_ID}")

# Choose data type based on hardware
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"Using dtype: {torch_dtype}")

if USE_QLORA:
    # QLoRA: Load model in 4-bit precision for memory efficiency
    print("Setting up QLoRA (4-bit quantization)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # Load model in 4-bit precision
        bnb_4bit_use_double_quant=True,       # Use nested quantization
        bnb_4bit_quant_type="nf4",           # Use NF4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,       # Apply 4-bit quantization
        device_map="auto",                    # Automatically place on GPU/CPU
        torch_dtype=torch_dtype,
    )
    
    # Prepare model for k-bit training (required for QLoRA)
    model = prepare_model_for_kbit_training(model)
    print("QLoRA setup complete")
    
else:
    # LoRA: Load model in full precision (16-bit)
    print("Setting up LoRA (16-bit precision)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    print("LoRA setup complete")

# ============================================================================
# LoRA CONFIGURATION SECTION
# ============================================================================
# Configure LoRA parameters (same for both LoRA and QLoRA)

print("Configuring LoRA adapter...")

# LoRA configuration
lora_cfg = LoraConfig(
    r=16,                                    # Rank of LoRA matrices (higher = more parameters)
    lora_alpha=32,                           # Scaling factor for LoRA weights
    target_modules=[                         # Which layers to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"      # MLP layers
    ],
    lora_dropout=0.05,                       # Dropout for LoRA layers
    bias="none",                             # Don't train bias terms
    task_type="CAUSAL_LM",                   # Task type for causal language modeling
)

# Apply LoRA configuration to model
model = get_peft_model(model, lora_cfg)

# Print trainable parameters info
model.print_trainable_parameters()
print("LoRA configuration complete")

# ============================================================================
# TRAINING SECTION
# ============================================================================
# Set up training arguments and start training

print("Setting up training...")

# Data collator for batching
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,                   # Where to save checkpoints
    per_device_train_batch_size=BATCH_SIZE,  # Batch size per GPU
    gradient_accumulation_steps=GRAD_ACCUM,  # Accumulate gradients over N steps
    num_train_epochs=EPOCHS,                 # Number of training epochs
    learning_rate=LR,                        # Learning rate
    fp16=not USE_QLORA,                      # Use fp16 for LoRA (not needed for QLoRA)
    bf16=True,                               # Use bfloat16 (better than fp16)
    logging_steps=LOG_STEPS,                 # Log every N steps
    save_steps=SAVE_STEPS,                   # Save checkpoint every N steps
    save_total_limit=2,                      # Keep only last 2 checkpoints
    warmup_steps=WARMUP_STEPS,               # Learning rate warmup
    lr_scheduler_type="cosine",              # Cosine learning rate schedule
    gradient_checkpointing=True,             # Save memory by recomputing gradients
    optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",  # Optimizer
    report_to=[],                            # Don't report to external services
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=collator,
)

print("Starting training...")
print(f"Training will save to: {OUTPUT_DIR}")

# Start training
trainer.train()

# ============================================================================
# SAVE MODEL SECTION
# ============================================================================
# Save the trained model and tokenizer

print("Training complete! Saving model...")

# Save the PEFT adapter (LoRA weights)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training complete! Adapter saved to: {OUTPUT_DIR}")
print(f"📁 Next steps:")
print(f"   1. Merge adapters: python merge_adapters.py")
print(f"   2. Test inference: python quantized_infer.py")
print(f"   3. Export to ONNX: python export_onnx.py")
print(f"   4. Serve with FastAPI: uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000")
