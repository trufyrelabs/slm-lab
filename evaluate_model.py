"""
SLM-Lab: Model Evaluation Script
===============================

This script helps you evaluate the performance of your fine-tuned model
by testing it with various prompts and analyzing the responses.

What this script does:
1. Loads your trained model
2. Runs a series of test prompts
3. Analyzes response quality and relevance
4. Provides metrics and insights
5. Saves evaluation results

Key Features:
- Multiple evaluation metrics
- Customizable test prompts
- Response quality analysis
- Performance benchmarking
- Detailed reporting

Usage:
    python evaluate_model.py

Environment Variables:
    MODEL_DIR: Path to model directory (default: ./outputs/tinyllama-merged)
    EVAL_OUTPUT: Output file for results (default: ./evaluation_results.json)
"""

import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any
import re

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# Model and evaluation configuration

MODEL_DIR = os.environ.get("MODEL_DIR", "./outputs/tinyllama-merged")
EVAL_OUTPUT = os.environ.get("EVAL_OUTPUT", "./evaluation_results.json")

# Test prompts for evaluation
TEST_PROMPTS = [
    {
        "category": "General Knowledge",
        "prompt": "Explain what machine learning is in simple terms.",
        "expected_keywords": ["artificial intelligence", "data", "patterns", "learn"]
    },
    {
        "category": "Code Generation",
        "prompt": "Write a Python function to calculate the factorial of a number.",
        "expected_keywords": ["def", "factorial", "return", "if", "else"]
    },
    {
        "category": "Summarization",
        "prompt": "Summarize the benefits of renewable energy in one paragraph.",
        "expected_keywords": ["sustainable", "environment", "clean", "energy"]
    },
    {
        "category": "Creative Writing",
        "prompt": "Write a short story about a robot learning to cook.",
        "expected_keywords": ["robot", "cook", "kitchen", "learn"]
    },
    {
        "category": "Problem Solving",
        "prompt": "How would you solve the problem of reducing plastic waste?",
        "expected_keywords": ["recycle", "reduce", "reuse", "plastic", "waste"]
    }
]

print(f"Model Evaluation Configuration:")
print(f"  Model Directory: {MODEL_DIR}")
print(f"  Output File: {EVAL_OUTPUT}")

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================
# Load the model and tokenizer for evaluation

print("Loading model for evaluation...")

# Set up 8-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    quantization_config=bnb_config
)

print(f"Model loaded successfully on {model.device}")

# ============================================================================
# EVALUATION FUNCTIONS SECTION
# ============================================================================
# Functions for evaluating model performance

def generate_response(prompt: str, max_tokens: int = 200) -> str:
    """
    Generate a response for a given prompt.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated response text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the original prompt from the response
    response = response.replace(prompt, "").strip()
    return response

def calculate_response_length(response: str) -> Dict[str, int]:
    """
    Calculate various length metrics for a response.
    
    Args:
        response: Generated response text
    
    Returns:
        Dictionary with length metrics
    """
    words = response.split()
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        "characters": len(response),
        "words": len(words),
        "sentences": len(sentences),
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0
    }

def check_keyword_presence(response: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    Check if expected keywords are present in the response.
    
    Args:
        response: Generated response text
        expected_keywords: List of expected keywords
    
    Returns:
        Dictionary with keyword analysis
    """
    response_lower = response.lower()
    found_keywords = []
    
    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            found_keywords.append(keyword)
    
    keyword_coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
    
    return {
        "expected_keywords": expected_keywords,
        "found_keywords": found_keywords,
        "keyword_coverage": keyword_coverage,
        "missing_keywords": [k for k in expected_keywords if k.lower() not in response_lower]
    }

def evaluate_response_quality(response: str) -> Dict[str, Any]:
    """
    Evaluate the quality of a generated response.
    
    Args:
        response: Generated response text
    
    Returns:
        Dictionary with quality metrics
    """
    # Basic quality checks
    has_content = len(response.strip()) > 0
    is_coherent = len(response.split()) > 5  # More than 5 words
    has_structure = any(char in response for char in ['.', '!', '?', '\n'])
    
    # Check for common issues
    is_repetitive = len(set(response.split())) / len(response.split()) < 0.7 if response.split() else False
    has_stop_words = any(word in response.lower() for word in ['the', 'and', 'or', 'but', 'in', 'on', 'at'])
    
    return {
        "has_content": has_content,
        "is_coherent": is_coherent,
        "has_structure": has_structure,
        "is_repetitive": is_repetitive,
        "has_stop_words": has_stop_words,
        "quality_score": sum([has_content, is_coherent, has_structure, has_stop_words, not is_repetitive]) / 5
    }

# ============================================================================
# MAIN EVALUATION SECTION
# ============================================================================
# Run the evaluation on all test prompts

print("\nStarting model evaluation...")
print("=" * 50)

evaluation_results = {
    "model_info": {
        "model_path": MODEL_DIR,
        "device": str(model.device),
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "test_results": [],
    "summary": {}
}

total_quality_score = 0
total_keyword_coverage = 0
total_generation_time = 0

for i, test_case in enumerate(TEST_PROMPTS, 1):
    print(f"\nTest {i}/{len(TEST_PROMPTS)}: {test_case['category']}")
    print(f"Prompt: {test_case['prompt']}")
    
    # Generate response and measure time
    start_time = time.time()
    response = generate_response(test_case['prompt'])
    generation_time = time.time() - start_time
    
    print(f"Response: {response[:100]}...")
    
    # Calculate metrics
    length_metrics = calculate_response_length(response)
    keyword_analysis = check_keyword_presence(response, test_case['expected_keywords'])
    quality_metrics = evaluate_response_quality(response)
    
    # Store results
    test_result = {
        "test_id": i,
        "category": test_case['category'],
        "prompt": test_case['prompt'],
        "response": response,
        "generation_time": generation_time,
        "length_metrics": length_metrics,
        "keyword_analysis": keyword_analysis,
        "quality_metrics": quality_metrics
    }
    
    evaluation_results["test_results"].append(test_result)
    
    # Update totals
    total_quality_score += quality_metrics['quality_score']
    total_keyword_coverage += keyword_analysis['keyword_coverage']
    total_generation_time += generation_time
    
    print(f"Quality Score: {quality_metrics['quality_score']:.2f}")
    print(f"Keyword Coverage: {keyword_analysis['keyword_coverage']:.2f}")
    print(f"Generation Time: {generation_time:.2f}s")

# ============================================================================
# SUMMARY SECTION
# ============================================================================
# Calculate overall performance metrics

print("\n" + "=" * 50)
print("EVALUATION SUMMARY")
print("=" * 50)

num_tests = len(TEST_PROMPTS)
avg_quality_score = total_quality_score / num_tests
avg_keyword_coverage = total_keyword_coverage / num_tests
avg_generation_time = total_generation_time / num_tests

evaluation_results["summary"] = {
    "total_tests": num_tests,
    "average_quality_score": avg_quality_score,
    "average_keyword_coverage": avg_keyword_coverage,
    "average_generation_time": avg_generation_time,
    "total_evaluation_time": total_generation_time
}

print(f"Total Tests: {num_tests}")
print(f"Average Quality Score: {avg_quality_score:.2f}/1.0")
print(f"Average Keyword Coverage: {avg_keyword_coverage:.2f}/1.0")
print(f"Average Generation Time: {avg_generation_time:.2f}s")
print(f"Total Evaluation Time: {total_generation_time:.2f}s")

# ============================================================================
# SAVE RESULTS SECTION
# ============================================================================
# Save evaluation results to file

print(f"\nSaving evaluation results to: {EVAL_OUTPUT}")

with open(EVAL_OUTPUT, 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("✅ Evaluation complete!")
print(f"📊 Results saved to: {EVAL_OUTPUT}")

# ============================================================================
# RECOMMENDATIONS SECTION
# ============================================================================
# Provide recommendations based on evaluation results

print("\n" + "=" * 50)
print("RECOMMENDATIONS")
print("=" * 50)

if avg_quality_score < 0.6:
    print("⚠️  Quality Score is low. Consider:")
    print("   - Training for more epochs")
    print("   - Improving your training data quality")
    print("   - Adjusting learning rate or batch size")

if avg_keyword_coverage < 0.5:
    print("⚠️  Keyword coverage is low. Consider:")
    print("   - Adding more examples with target keywords")
    print("   - Reviewing your training data diversity")
    print("   - Adjusting the prompt format")

if avg_generation_time > 5.0:
    print("⚠️  Generation is slow. Consider:")
    print("   - Using a smaller model")
    print("   - Reducing max_new_tokens")
    print("   - Using GPU acceleration")

if avg_quality_score >= 0.8 and avg_keyword_coverage >= 0.7:
    print("🎉 Excellent model performance!")
    print("   Your model is ready for production use.")

print("\n💡 Next steps:")
print("   1. Review individual test results in the JSON file")
print("   2. Iterate on training data based on weak areas")
print("   3. Test with domain-specific prompts")
print("   4. Deploy your model: python serve_fastapi.py")
