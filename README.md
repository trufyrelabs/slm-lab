# slm-lab

> Practical toolkit for fine-tuning, optimizing, and deploying Small Language Models for focused tasks.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [1. Choose a Base Model](#1-choose-a-base-model)
  - [2. Prepare Your Dataset](#2-prepare-your-dataset)
  - [3. Fine-tune with LoRA/QLoRA](#3-fine-tune-with-loraqlora)
  - [4. Merge Adapters](#4-merge-adapters)
  - [5. Test Inference](#5-test-inference)
  - [6. Export to ONNX](#6-export-to-onnx)
  - [7. Deploy with FastAPI](#7-deploy-with-fastapi)
  - [8. Docker Deployment](#8-docker-deployment)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Model Selection Guide](#model-selection-guide)
- [Data Preparation Tips](#data-preparation-tips)
- [Deployment Options](#deployment-options)
- [FAQ](#faq)
- [Contributing](#contributing)

## 🚀 Overview

SLM-Lab is a comprehensive toolkit for training domain-specific Small Language Models efficiently. Perfect for:

- **Customer Support Bots** - Automated responses with company knowledge
- **Internal QA Assistants** - Domain-specific question answering
- **Document Summarization** - Contract, medical, or legal document processing
- **Data Extraction** - Structured information extraction from text
- **Focused Task Automation** - Any specialized language task

### Key Benefits

✅ **Efficient Training** - LoRA/QLoRA for memory-efficient fine-tuning  
✅ **Multiple Base Models** - Support for TinyLlama, Phi-2, Mistral, Llama-3.x  
✅ **Optimization Ready** - Quantization and ONNX export included  
✅ **Production Ready** - FastAPI server with Docker support  
✅ **Edge Deployment** - ONNX models for on-device inference  

## ✨ Features

- **LoRA/QLoRA Fine-tuning** - Memory-efficient adapter training
- **Multiple Base Models** - Easy switching between open-source models
- **Quantization Support** - 8-bit inference and INT8 ONNX export
- **FastAPI Server** - Production-ready inference API
- **Docker Support** - Containerized deployment
- **ONNX Export** - Cross-platform model deployment
- **Flexible Data Format** - JSONL-based instruction tuning

## 🏃 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -U "transformers>=4.41" datasets accelerate peft bitsandbytes \
  trl optimum onnx onnxruntime onnxruntime-gpu fastapi uvicorn pydantic==2.* einops safetensors
```

### Basic Workflow

```bash
# 1. Prepare your data
cp your_data.jsonl ./data/instructions.jsonl

# 2. Train with QLoRA (default)
python train_lora.py

# 3. Merge adapters
python merge_adapters.py

# 4. Test inference
python quantized_infer.py

# 5. Evaluate model performance
python evaluate_model.py

# 6. Export to ONNX
python export_onnx.py

# 7. Serve with FastAPI
uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000
```

## 📋 Requirements

### System Requirements

- **Python**: 3.10+
- **GPU**: NVIDIA with CUDA (recommended for training)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 5GB+ free space
- **OS**: Linux, macOS, or Windows

### Hardware Recommendations

| Task | Minimum | Recommended |
|------|---------|-------------|
| Training (QLoRA) | 8GB VRAM | 16GB+ VRAM |
| Training (LoRA) | 12GB VRAM | 24GB+ VRAM |
| Inference | 4GB VRAM | 8GB+ VRAM |
| CPU Training | 16GB RAM | 32GB+ RAM |

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/trufyrelabs/slm-lab.git
   cd slm-lab
   ```

2. **Install dependencies**
   ```bash
   pip install -U "transformers>=4.41" datasets accelerate peft bitsandbytes \
     trl optimum onnx onnxruntime onnxruntime-gpu fastapi uvicorn pydantic==2.* einops safetensors
   ```

3. **Verify installation**
   ```bash
   python -c "import transformers, peft, bitsandbytes; print('Installation successful!')"
   ```

## 📖 Usage Guide

### 1. Choose a Base Model

Set `BASE_MODEL_ID` to any open-source model on Hugging Face:

```bash
# Lightweight starter
export BASE_MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Medium size
export BASE_MODEL_ID="microsoft/phi-2"

# Larger model
export BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"

# Gated model (requires access)
export BASE_MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
```

**Recommended Models by Use Case:**

| Use Case | Model | Size | Speed | Quality |
|----------|-------|------|-------|---------|
| Quick prototyping | TinyLlama-1.1B | 1.1B | ⚡⚡⚡ | ⭐⭐ |
| General purpose | Phi-2 | 2.7B | ⚡⚡ | ⭐⭐⭐ |
| High quality | Mistral-7B | 7B | ⚡ | ⭐⭐⭐⭐ |
| Production | Llama-3.2-3B | 3.2B | ⚡⚡ | ⭐⭐⭐⭐ |

### 2. Prepare Your Dataset

Create `./data/instructions.jsonl` with one JSON per line:

```json
{"instruction": "Answer the company FAQ", "input": "What is our refund policy?", "output": "Refunds are available within 30 days with proof of purchase."}
{"instruction": "Summarize the clause", "input": "The tenant must...", "output": "Tenant responsible for utilities; 30-day notice to terminate."}
{"instruction": "Extract fields from invoice", "input": "Invoice #49382 dated 2024-10-05 total $1,240", "output": "{\"invoice_no\":\"49382\",\"date\":\"2024-10-05\",\"total\":1240}"}
```

**Dataset Format:**
- `instruction` (required): What the model should do
- `input` (optional): Additional context
- `output` (required): Expected response

**Data Quality Tips:**
- Aim for 1,000-10,000 high-quality examples
- Keep instructions clear and consistent
- Avoid contradictory examples
- Use diverse but focused scenarios

### 3. Fine-tune with LoRA/QLoRA

**QLoRA (Recommended - Memory Efficient):**
```bash
export USE_QLORA=1
python train_lora.py
```

**LoRA (Traditional - More Memory):**
```bash
export USE_QLORA=0
python train_lora.py
```

**Custom Training Parameters:**
```bash
export BATCH_SIZE=2 GRAD_ACCUM=8 EPOCHS=2 LR=2e-4 MAX_SEQ_LEN=1024
python train_lora.py
```

**Output:** `./outputs/tinyllama-lora/`

### 4. Merge Adapters

Merge LoRA adapters into the base model for simpler deployment:

```bash
python merge_adapters.py
```

**Output:** `./outputs/tinyllama-merged/`

### 5. Test Inference

Quick sanity check with 8-bit loading:

```bash
python quantized_infer.py
```

### 6. Evaluate Model Performance

Comprehensive evaluation with multiple test cases:

```bash
python evaluate_model.py
```

**What the evaluation does:**
- Tests model on various prompt types (general knowledge, code generation, summarization, etc.)
- Analyzes response quality, keyword coverage, and generation speed
- Provides detailed metrics and recommendations
- Saves results to `evaluation_results.json`

**Sample output:**
```
Test 1/5: General Knowledge
Prompt: Explain what machine learning is in simple terms.
Response: Machine learning is a subset of artificial intelligence...
Quality Score: 0.85
Keyword Coverage: 0.75
Generation Time: 2.34s

EVALUATION SUMMARY
Total Tests: 5
Average Quality Score: 0.82/1.0
Average Keyword Coverage: 0.78/1.0
Average Generation Time: 2.15s
```

### 7. Export to ONNX

Export for CPU/edge deployment with INT8 quantization:

```bash
python export_onnx.py
```

**Outputs:**
- `./onnx_export/model.onnx` (FP32)
- `./onnx_export/model-int8.onnx` (INT8 quantized)

### 8. Deploy with FastAPI

Start the inference server:

```bash
uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000
```

**API Usage:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain SLMs vs LLMs briefly.",
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

### 9. Docker Deployment

**Build and run:**
```bash
docker build -t slm-api .
docker run --gpus all -p 8000:8000 -e MODEL_DIR=/app/outputs/tinyllama-merged slm-api
```

**With custom model:**
```bash
docker run --gpus all -p 8000:8000 \
  -v $PWD/outputs/tinyllama-merged:/models/merged \
  -e MODEL_DIR=/models/merged slm-api
```

## 📁 Project Structure

```
slm-lab/
├── README.md                # This file
├── train_lora.py            # LoRA/QLoRA fine-tuning
├── merge_adapters.py        # Merge LoRA adapters
├── quantized_infer.py       # 8-bit inference test
├── evaluate_model.py        # Model evaluation & testing
├── export_onnx.py           # ONNX export & quantization
├── serve_fastapi.py         # FastAPI inference server
├── Dockerfile               # GPU-ready Docker image
├── requirements.txt         # Python dependencies
├── data/
│   └── instructions.jsonl   # Training dataset
├── outputs/                 # Training outputs
│   ├── tinyllama-lora/      # LoRA adapters
│   └── tinyllama-merged/    # Merged model
├── onnx_export/             # ONNX models
│   ├── model.onnx
│   └── model-int8.onnx
└── evaluation_results.json  # Model evaluation results
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_MODEL_ID` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base model to fine-tune |
| `USE_QLORA` | `1` | Use QLoRA (1) or LoRA (0) |
| `BATCH_SIZE` | `1` | Training batch size |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps |
| `EPOCHS` | `3` | Number of training epochs |
| `LR` | `2e-4` | Learning rate |
| `MAX_SEQ_LEN` | `512` | Maximum sequence length |
| `MODEL_DIR` | `./outputs/tinyllama-merged` | Model directory for serving |

### Training Parameters

| Parameter | QLoRA | LoRA | Description |
|-----------|-------|------|-------------|
| Memory Usage | ~8GB | ~16GB | VRAM requirements |
| Training Speed | Fast | Medium | Relative training time |
| Model Quality | High | High | Final model performance |
| Compatibility | Limited | Broad | Hardware requirements |

## 🔧 Troubleshooting

### Common Issues

**1. bitsandbytes errors on CPU**
```bash
# Solution: Use LoRA instead of QLoRA
export USE_QLORA=0
python train_lora.py
```

**2. CUDA Out of Memory**
```bash
# Reduce memory usage
export BATCH_SIZE=1 GRAD_ACCUM=8 MAX_SEQ_LEN=256
python train_lora.py
```

**3. Gated model access**
- Request access on Hugging Face first
- Use `huggingface-cli login` to authenticate

**4. Tokenizer padding errors**
- Update transformers: `pip install -U transformers`
- Script automatically sets `pad_token=EOS` if missing

**5. Slow CPU training**
- Use smaller models (TinyLlama, Phi-2)
- Consider cloud GPU instances
- Reduce dataset size for testing

### Performance Optimization

| Optimization | Memory | Speed | Quality |
|--------------|--------|-------|---------|
| Reduce batch size | ⬇️ | ⬇️ | ➡️ |
| Increase grad accum | ⬇️ | ⬇️ | ➡️ |
| Reduce seq length | ⬇️ | ⬆️ | ⬇️ |
| Use QLoRA | ⬇️ | ⬆️ | ➡️ |
| Quantize inference | ⬇️ | ⬆️ | ⬇️ |

## 🎯 Model Selection Guide

### Quick Decision Tree

```
Start Here
├── Need fast prototyping?
│   └── TinyLlama-1.1B (1.1B params)
├── Have 8GB VRAM?
│   └── Phi-2 (2.7B params)
├── Have 16GB+ VRAM?
│   └── Mistral-7B (7B params)
└── Need production quality?
    └── Llama-3.2-3B (3.2B params)
```

### Model Comparison

| Model | Params | VRAM (QLoRA) | VRAM (LoRA) | Speed | Quality |
|-------|--------|--------------|-------------|-------|---------|
| TinyLlama-1.1B | 1.1B | 4GB | 6GB | ⚡⚡⚡ | ⭐⭐ |
| Phi-2 | 2.7B | 6GB | 10GB | ⚡⚡ | ⭐⭐⭐ |
| Mistral-7B | 7B | 12GB | 20GB | ⚡ | ⭐⭐⭐⭐ |
| Llama-3.2-3B | 3.2B | 8GB | 14GB | ⚡⚡ | ⭐⭐⭐⭐ |

## 📊 Data Preparation Tips

### Dataset Guidelines

**Size Recommendations:**
- **Minimum**: 500 examples
- **Recommended**: 1,000-10,000 examples
- **Optimal**: 5,000-50,000 examples

**Quality Guidelines:**
- ✅ Clear, unambiguous instructions
- ✅ Consistent formatting
- ✅ Diverse scenarios
- ✅ High-quality outputs
- ❌ Contradictory examples
- ❌ Ambiguous instructions
- ❌ Low-quality responses

**Format Examples:**

**QA Assistant:**
```json
{"instruction": "Answer the question", "input": "What is machine learning?", "output": "Machine learning is a subset of artificial intelligence..."}
```

**Summarization:**
```json
{"instruction": "Summarize the text", "input": "Long document text...", "output": "Brief summary of key points..."}
```

**Code Generation:**
```json
{"instruction": "Write a Python function", "input": "Create a function that sorts a list", "output": "def sort_list(lst):\n    return sorted(lst)"}
```

### Model Evaluation

After training, evaluate your model's performance:

```bash
python evaluate_model.py
```

**Evaluation Metrics:**
- **Quality Score**: Overall response quality (0-1)
- **Keyword Coverage**: Presence of expected keywords (0-1)
- **Generation Time**: Response generation speed
- **Response Length**: Character, word, and sentence counts

**Test Categories:**
- General Knowledge
- Code Generation
- Summarization
- Creative Writing
- Problem Solving

**Customization:**
- Modify `TEST_PROMPTS` in `evaluate_model.py` for domain-specific testing
- Adjust quality metrics based on your use case
- Add custom evaluation criteria

## 🚀 Deployment Options

### 1. FastAPI Server (Recommended)

**Pros:** Easy setup, production-ready, Docker support  
**Cons:** Requires server infrastructure  

```bash
uvicorn serve_fastapi:app --host 0.0.0.0 --port 8000
```

### 2. ONNX Runtime

**Pros:** Cross-platform, CPU/GPU support, edge deployment  
**Cons:** Limited model compatibility  

```python
import onnxruntime as ort
session = ort.InferenceSession("model-int8.onnx")
```

### 3. llama.cpp (GGUF)

**Pros:** Excellent CPU performance, mobile support  
**Cons:** Requires model conversion  

```bash
# Convert to GGUF format
python -m llama_cpp.convert_model ./outputs/tinyllama-merged --outfile model.gguf
```

### 4. MLC-LLM

**Pros:** Mobile/edge deployment, optimized compilation  
**Cons:** Complex setup, limited model support  

### 5. ONNX Runtime Mobile

**Pros:** Cross-platform mobile deployment  
**Cons:** Limited features compared to desktop  

## ❓ FAQ

**Q: Can I use CSV/Parquet instead of JSONL?**  
A: Yes! Load with `datasets` and map columns to instruction/input/output fields.

**Q: How do I change the prompt template?**  
A: Edit the `format_example()` function in `train_lora.py`.

**Q: Can I use multiple GPUs?**  
A: Install `accelerate` and run with distributed training support.

**Q: What's the difference between LoRA and QLoRA?**  
A: QLoRA uses 4-bit quantization during training, reducing memory usage by ~50%.

**Q: How do I evaluate my model?**  
A: Use the built-in evaluation script: `python evaluate_model.py`. It tests multiple scenarios and provides quality metrics, keyword coverage, and performance recommendations.

**Q: Can I fine-tune on CPU?**  
A: Yes, but it will be very slow. Use small models and limited data for testing.

**Q: How do I handle sensitive data?**  
A: Keep data local, use appropriate licenses, and maintain human oversight for critical applications.

## 🤝 Contributing

We welcome contributions! Here are some areas where you can help:

### Code Contributions
- Add evaluation scripts
- Improve error handling
- Add new model support
- Optimize performance

### Documentation
- Add deployment recipes
- Create tutorials
- Improve examples
- Add troubleshooting guides

### Data & Models
- Share dataset examples
- Add prompt templates
- Contribute model configurations
- Create evaluation benchmarks

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon the amazing open-source ecosystem:

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [Optimum](https://github.com/huggingface/optimum)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [FastAPI](https://github.com/tiangolo/fastapi)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/trufyrelabs/slm-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/trufyrelabs/slm-lab/discussions)
- **Documentation**: [Wiki](https://github.com/trufyrelabs/slm-lab/wiki)

---

**Happy building! 🚀**

If you build something cool with this toolkit, consider sharing it in our [Discussions](https://github.com/trufyrelabs/slm-lab/discussions) so others can learn from your work!