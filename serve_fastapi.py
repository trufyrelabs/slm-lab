"""
SLM-Lab: FastAPI Inference Server
================================

This script creates a FastAPI web server for serving your trained model via HTTP API.
It provides a simple REST API for text generation that can be used in production
environments or integrated with other applications.

What this script does:
1. Loads the trained model (merged format recommended)
2. Creates a FastAPI web server
3. Provides a /generate endpoint for text generation
4. Supports streaming responses (basic implementation)
5. Handles concurrent requests efficiently

Key Features:
- RESTful API for text generation
- Configurable generation parameters
- Streaming support for long responses
- Production-ready with proper error handling
- Easy integration with web applications

Usage:
    uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /generate - Generate text from a prompt
    GET /health - Health check endpoint

Environment Variables:
    MODEL_DIR: Path to model directory (default: ./outputs/tinyllama-merged)
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
from typing import Optional

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# Model and server configuration

# Model directory - change this to point to your trained model
MODEL_DIR = os.environ.get("MODEL_DIR", "./outputs/tinyllama-merged")

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"FastAPI Server Configuration:")
print(f"  Model Directory: {MODEL_DIR}")
print(f"  Device: {DEVICE}")

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================
# Load the model and tokenizer for inference

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

print("Loading model...")
# Load the model with appropriate precision for the device
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
)
model.to(DEVICE)
model.eval()  # Set to evaluation mode

print(f"Model loaded successfully on {DEVICE}")

# ============================================================================
# FASTAPI APPLICATION SECTION
# ============================================================================
# Create the FastAPI application

app = FastAPI(
    title="SLM Inference API",
    description="API for generating text using fine-tuned Small Language Models",
    version="1.0.0"
)

# ============================================================================
# REQUEST/RESPONSE MODELS SECTION
# ============================================================================
# Define the data models for API requests and responses

class GenRequest(BaseModel):
    """
    Request model for text generation.
    
    Attributes:
        prompt: The input text to generate from
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
        top_p: Nucleus sampling parameter (0.0 to 1.0)
        do_sample: Whether to use sampling or greedy decoding
    """
    prompt: str = Field(..., description="Input prompt for text generation")
    max_new_tokens: int = Field(default=256, ge=1, le=2048, description="Maximum new tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    do_sample: bool = Field(default=True, description="Use sampling instead of greedy decoding")

class GenResponse(BaseModel):
    """
    Response model for text generation.
    
    Attributes:
        output: The generated text
        prompt: The original input prompt
        parameters: The generation parameters used
    """
    output: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original input prompt")
    parameters: dict = Field(..., description="Generation parameters used")

# ============================================================================
# API ENDPOINTS SECTION
# ============================================================================
# Define the API endpoints

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the server is running.
    
    Returns:
        Dictionary with server status and model information
    """
    return {
        "status": "healthy",
        "model": MODEL_DIR,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=GenResponse)
async def generate(req: GenRequest):
    """
    Generate text from a prompt.
    
    This endpoint takes a text prompt and generates a response using the
    fine-tuned model. It supports various generation parameters for
    controlling the output quality and style.
    
    Args:
        req: Generation request with prompt and parameters
    
    Returns:
        Generated text response with metadata
    
    Raises:
        HTTPException: If generation fails or parameters are invalid
    """
    try:
        # Tokenize the input prompt
        inputs = tok(req.prompt, return_tensors="pt").to(DEVICE)
        
        # Set up streaming for response generation
        streamer = TextIteratorStreamer(
            tok, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Prepare generation parameters
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            temperature=req.temperature,
            top_p=req.top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id
        )
        
        # Generate text in a separate thread (required for streaming)
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Collect the generated text
        # Note: This is a simplified streaming implementation
        # For production, consider using Server-Sent Events (SSE) or WebSockets
        result = "".join(list(streamer))
        
        # Wait for the generation thread to complete
        thread.join()
        
        # Return the response
        return GenResponse(
            output=result,
            prompt=req.prompt,
            parameters={
                "max_new_tokens": req.max_new_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "do_sample": req.do_sample
            }
        )
        
    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(
            status_code=500, 
            detail=f"Generation failed: {str(e)}"
        )

# ============================================================================
# ADDITIONAL ENDPOINTS SECTION
# ============================================================================
# Optional: Add more endpoints for different use cases

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary with model configuration and statistics
    """
    return {
        "model_path": MODEL_DIR,
        "device": DEVICE,
        "vocab_size": tok.vocab_size,
        "model_type": model.config.model_type,
        "max_position_embeddings": model.config.max_position_embeddings,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

# ============================================================================
# STARTUP/SHUTDOWN EVENTS SECTION
# ============================================================================
# Handle application startup and shutdown

@app.on_event("startup")
async def startup_event():
    """Called when the FastAPI application starts."""
    print("🚀 SLM Inference API server starting...")
    print(f"📁 Model: {MODEL_DIR}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"🌐 Server will be available at: http://localhost:8000")
    print(f"📚 API docs will be available at: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Called when the FastAPI application shuts down."""
    print("🛑 SLM Inference API server shutting down...")

# ============================================================================
# USAGE EXAMPLES SECTION
# ============================================================================
"""
Example usage of the FastAPI server:

1. Start the server:
   ```bash
   uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000
   ```

2. Generate text using curl:
   ```bash
   curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Explain machine learning in simple terms.",
       "max_new_tokens": 200,
       "temperature": 0.7,
       "top_p": 0.9
     }'
   ```

3. Generate text using Python:
   ```python
   import requests
   
   response = requests.post(
       "http://localhost:8000/generate",
       json={
           "prompt": "Write a short story about a robot.",
           "max_new_tokens": 150,
           "temperature": 0.8
       }
   )
   
   result = response.json()
   print(result["output"])
   ```

4. Check server health:
   ```bash
   curl http://localhost:8000/health
   ```

5. View API documentation:
   Open http://localhost:8000/docs in your browser

Customization options:
- Change MODEL_DIR environment variable to use different models
- Modify generation parameters in the GenRequest model
- Add authentication middleware for production use
- Implement rate limiting for API protection
- Add logging and monitoring endpoints
- Customize error handling and response formats
"""
