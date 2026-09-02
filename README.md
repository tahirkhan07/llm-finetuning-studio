# LLM Fine-Tuning Studio

A professional, modular desktop environment for crafting, evaluating, and serving fine-tuned Large Language Models. Built with Gradio, PyTorch, Hugging Face `transformers`, `peft`, and `trl`.

This studio provides an end-to-end graphical interface to perform **QLoRA** (Quantized Low-Rank Adaptation) supervised fine-tuning directly on your local hardware.

## ✨ Features

- **Any Model Support:** Paste any Hugging Face model ID into the Model tab — works with Qwen, Llama, Gemma, Mistral, Phi, and thousands more.
- **Automatic Prompt Formatting:** Each model's tokenizer ships with its own `chat_template`. The studio applies it automatically, eliminating the need to manually format prompts.
- **Hardware Awareness:** Automatically detects your GPU, estimates VRAM requirements, and recommends optimal training hyperparameters.
- **Dataset Management:** Seamlessly map CSV, JSONL, or Hugging Face dataset columns to a canonical conversation format, including token-length validation before training.
- **Unified Training Engine:** Supports QLoRA and standard LoRA with live, streaming loss curves displayed directly in the UI.
- **Evaluation & Inference:** Chat with your fine-tuned model, apply domain guardrails, calculate perplexity, and benchmark generation speed side-by-side with the base model.
- **Model Export & Publishing:** Easily export and convert your fine-tuned adapters into merged base models. Push your models and adapters directly to the Hugging Face Hub straight from the UI.
- **Experiment Tracking:** Systematically logs all training runs, configurations, metrics, and adapter paths for easy comparison.

## 🛠️ Requirements

- **Python:** ≥ 3.10
- **CUDA:** ≥ 11.8 (for GPU training)
- **NVIDIA GPU:** Sufficient VRAM (8 GB+ recommended for 1–3B models in 4-bit precision)

## 📦 Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast and reliable Python dependency management. You do not need to manually create a virtual environment or install packages — `uv` handles everything automatically.

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/llm-finetuning-studio.git
   cd llm-finetuning-studio
   ```

3. **Launch the Studio**:
   ```bash
   uv run python app/main.py
   ```
   *Note: On the first run, `uv` will automatically create a `.venv`, resolve dependencies, and download PyTorch and CUDA binaries. This may take a few minutes.*

4. **Access the UI**:
   Open your web browser and navigate to [http://127.0.0.1:7860](http://127.0.0.1:7860).

## 🧪 Testing

To run the unit and integration tests:
```bash
uv run --with pytest --with pytest-mock pytest tests/
```

## ⚠️ Disclaimer

This project is intended for research and educational purposes. Always review the license and usage conditions of the base models and datasets you use.
