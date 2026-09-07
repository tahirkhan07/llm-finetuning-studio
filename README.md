# LLM Fine-Tuning Studio

A local, end-to-end platform for fine-tuning, evaluating, and serving LLMs using LoRA and QLoRA.
Built with FastAPI, PyTorch, Hugging Face Transformers, PEFT, and TRL.

## ✨ Features

### 🧠 Model Management
- Hugging Face and local model support
- Automatic chat-template detection
- Automatic LoRA target-module detection
- Adapter management

### 📊 Dataset Pipeline
- Hugging Face, CSV, and JSONL datasets
- Flexible column mapping
- Configurable Train / Validation / Test splits
- Max Training Samples
- Exact duplicate removal
- Context-length validation

### 🏋️ Fine-Tuning
- LoRA and QLoRA
- 4-bit quantization
- GPU and VRAM detection
- Gradient accumulation
- Live training and validation loss
- Best-checkpoint selection
- Early stopping
- Configurable evaluation/checkpoint frequency

### 📈 Evaluation
Evaluation is performed exclusively on the held-out test set.
The base model and fine-tuned model are evaluated on the same test samples using identical generation settings.

| Metric | Purpose |
|--------|---------|
| Perplexity | Language-model quality |
| ROUGE-L | Text overlap |
| BLEU | N-gram overlap |
| BERTScore | Semantic similarity |
| Token F1 | Answer-level token overlap |
| LLM Correctness | Correctness score (1–5) |
| LLM Relevance | Relevance score (1–5) |
| LLM Completeness | Completeness score (1–5) |
| Latency | Response speed |
| Tokens/sec | Generation throughput |

Results are shown Base vs Fine-tuned to measure the impact of fine-tuning.

### 💬 Inference
- Interactive multi-turn chat
- Streaming generation
- Adapter switching
- Optional safety guardrails

### 📦 Export
- Merge LoRA adapters
- Export trained models
- GGUF conversion
- Hugging Face Hub publishing

### 🔬 Experiment Tracking
- Training configuration and metrics
- Checkpoint and adapter tracking
- Experiment comparison
- Adapter downloads

## 🏗️ Architecture

```text
Dataset 
  ↓ 
Validation / Cleaning 
  ↓ 
Deduplication 
  ↓ 
Train / Validation / Test Split 
  ↓ 
Max Training Samples 
  ↓ 
LoRA / QLoRA Training 
  ├── Train → Update weights 
  └── Validation → Monitor / Select checkpoint 
  ↓ 
Best Checkpoint 
  ↓ 
Held-Out Test Set 
  ↓ 
Base vs Fine-Tuned 
  ↓ 
Inference
```

## 🛠️ Tech Stack
- **Backend:** FastAPI
- **Deep Learning:** PyTorch
- **Models:** Hugging Face Transformers
- **Fine-Tuning:** PEFT + TRL
- **Quantization:** BitsAndBytes
- **Datasets:** Hugging Face Datasets
- **Evaluation:** ROUGE, BLEU, BERTScore, Token F1
- **LLM Judge:** Gemini
- **Frontend:** HTML / CSS / JavaScript
- **Package Management:** uv
- **GPU:** CUDA

## 💻 Requirements
- Python >= 3.10
- NVIDIA GPU recommended
- CUDA >= 11.8 for GPU training
- 8 GB+ VRAM recommended for smaller models with QLoRA
- Git
- [uv](https://docs.astral.sh/uv/)

## 🚀 Setup & Run

### 1. Clone Repository
```bash
git clone https://github.com/tahirkhan07/llm-finetuning-studio.git 
cd llm-finetuning-studio
```

### 2. Install Dependencies
Install all project dependencies using uv:
```bash
uv sync
```

### 3. Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```
Update `.env` with the required configuration.
Example:
```
GEMINI_API_KEY=your_api_key
```
*The Gemini API key is required only for LLM-as-a-Judge evaluation.*

### 4. Verify GPU
Check NVIDIA GPU availability:
```bash
nvidia-smi
```
Verify PyTorch CUDA support:
```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 5. Start the Application
```bash
uv run python app/main.py
```
Open the application in your browser:
http://127.0.0.1:8000

## 🔄 Typical Workflow
```text
Select Model 
  ↓ 
Upload Dataset 
  ↓ 
Map Dataset Columns 
  ↓ 
Configure Train / Val / Test 
  ↓ 
Configure LoRA / QLoRA 
  ↓ 
Start Training 
  ↓ 
Monitor Train / Validation Loss 
  ↓ 
Best Checkpoint 
  ↓ 
Evaluate Held-Out Test Set 
  ↓ 
Base vs Fine-Tuned Comparison 
  ↓ 
Inference 
  ↓ 
Export / Publish
```

## 🧪 Testing
Run the test suite:
```bash
uv run --with pytest --with pytest-mock pytest tests/
```

## 📁 Project Structure
```text
llm-finetuning-studio/
├── app/
│   ├── main.py
│   ├── state.py
│   └── api/
│       └── routers/
│   ├── core/
│   ├── datasets/       # Loading, mapping, validation, splitting
│   ├── evaluation/     # Evaluation metrics
│   ├── hardware/       # GPU detection and planning
│   ├── inference/      # Model loading and generation
│   ├── models/         # Model inspection and export
│   └── training/       # LoRA / QLoRA / SFT
├── frontend/           # Web interface
├── tests/              # Tests
├── pyproject.toml
├── .env.example
└── README.md
```

