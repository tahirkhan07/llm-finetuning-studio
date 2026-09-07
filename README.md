# LLM Fine-Tuning Studio

A professional, end-to-end platform for fine-tuning, evaluating, and serving Large Language Models. Built with **FastAPI**, **PyTorch**, Hugging Face `transformers`, `peft`, and `trl`.

Provides a complete web-based graphical interface to perform **QLoRA** and **LoRA** supervised fine-tuning directly on your local hardware — no cloud required.

## ✨ Features

### 🧠 Model Management
- **Any Model Support** — Paste any Hugging Face model ID: works with Qwen, Llama, Gemma, Mistral, Phi, and thousands more.
- **Automatic Prompt Formatting** — Each model's `chat_template` is detected and applied automatically.
- **LoRA Target Detection** — Automatically identifies optimal LoRA target modules for each architecture.

### 📊 Dataset Pipeline
- **Flexible Loading** — Load datasets from Hugging Face Hub, CSV, or JSONL files.
- **Column Mapping** — Map any column structure to a canonical conversation format (instruction, input, output, system prompt).
- **Configurable Train/Val/Test Splits** — Set custom split ratios (e.g. 80/10/10) directly from the UI.
- **Max Training Samples** — Limit training data independently without affecting validation or test set sizes.
- **Deduplication** — Automatic exact-duplicate detection before splitting to prevent data leakage.

### 🏋️ Training Engine
- **QLoRA & LoRA** — Supports both 4-bit quantized and full-precision fine-tuning.
- **Live Training Logs** — Real-time streaming of training loss, eval loss, learning rate, and epoch.
- **Validation Tracking** — Passes a held-out validation set to `SFTTrainer` for periodic `eval_loss` monitoring.
- **Best Checkpoint Selection** — Automatically loads the checkpoint with the lowest validation loss (`load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`).
- **Early Stopping** — Configurable patience-based early stopping to prevent overfitting.
- **Configurable Eval/Save Frequency** — Set evaluation and checkpoint intervals (in steps) from the UI.

### 📈 Evaluation (Test Set Only)
The test set is **never** used during training. It is reserved exclusively for final evaluation.

| Metric | Description |
|---|---|
| Perplexity | Token-weighted cross-entropy loss |
| ROUGE-L | Longest common subsequence overlap |
| BLEU | N-gram precision |
| BERTScore | Semantic similarity via embeddings |
| Task Accuracy (F1) | Token-overlap F1 score |
| Response Latency | Average inference time per sample |
| Tokens/sec | Generation throughput |
| LLM Correctness | Gemini-judged factual accuracy (1–5) |
| LLM Relevance | Gemini-judged prompt adherence (1–5) |
| LLM Completeness | Gemini-judged answer completeness (1–5) |

All metrics are computed for both the **base model** and the **fine-tuned model** side-by-side.

### 💬 Inference
- **Interactive Chat** — Multi-turn conversation with your fine-tuned model.
- **Adapter Switching** — Load any trained adapter or use the base model.
- **Streaming Output** — Token-by-token streaming for responsive chat.
- **Safety Guardrails** — Optional content filtering.

### 📦 Export & Publishing
- **Merge & Download** — Merge LoRA adapters into the base model and download as a ZIP.
- **GGUF Conversion** — Convert merged models to GGUF format for llama.cpp / Ollama.
- **Push to Hub** — Publish models directly to Hugging Face Hub.

### 🔬 Experiment Tracking
- Logs all training runs with full configuration, final loss, and adapter paths.
- Compare any two experiments side-by-side.
- Download individual adapter weights.

## 🏗️ Architecture

```
Raw Dataset
    ↓
Cleaning / Validation
    ↓
Deduplication
    ↓
Train / Val / Test Split
    ↓
Max Training Samples (train only)
    ↓
       ┌───────────────┐
       │   SFT/QLoRA   │
       │               │
       │ Train → learn │
       │ Val → monitor │
       └───────────────┘
              ↓
       Best Checkpoint
              ↓
       Final Test Set
              ↓
       Base vs Fine-tuned
```

## 🛠️ Requirements

- **Python:** ≥ 3.10
- **CUDA:** ≥ 11.8 (for GPU training)
- **NVIDIA GPU:** 8 GB+ VRAM recommended (for 1–3B models in 4-bit)

## 📦 Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/tahirkhan07/llm-finetuning-studio.git
   cd llm-finetuning-studio
   ```

3. **Configure environment** (optional, for LLM-as-a-judge):
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

4. **Launch the Studio**:
   ```bash
   uv run python app/main.py
   ```
   > On the first run, `uv` will automatically create a `.venv`, resolve all dependencies, and download PyTorch + CUDA binaries. This may take a few minutes.

5. **Open the UI**:
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 📁 Project Structure

```
├── app/                    # FastAPI application
│   ├── main.py             # Server entrypoint
│   ├── state.py            # Global application state
│   └── api/routers/        # API route handlers
├── core/                   # Core ML logic
│   ├── datasets/           # Loading, mapping, splitting, deduplication
│   ├── evaluation/         # Metrics (perplexity, ROUGE, BLEU, LLM-judge)
│   ├── hardware/           # GPU detection & config recommendation
│   ├── inference/          # Model loading & text generation
│   ├── models/             # Model inspection, adapters, export
│   └── training/           # SFT & QLoRA engines, callbacks, config
├── frontend/               # Web UI (HTML + CSS + JS)
├── tests/                  # Unit & integration tests
├── pyproject.toml          # Project dependencies
└── .env                    # Environment variables (not committed)
```

## 🧪 Testing

```bash
uv run --with pytest --with pytest-mock pytest tests/
```

## ⚠️ Disclaimer

This project is intended for research and educational purposes. Always review the license and usage conditions of the base models and datasets you use.
