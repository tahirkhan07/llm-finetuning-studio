"""
Unified QLoRA Dashboard — Gradio Blocks UI
Covers: Training, Evaluation, and Inference in a single app.
"""

import sys
import os
import threading
import torch

import gradio as gr

# Add src/ to path so relative imports work when launched from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ACTIVE_DOMAIN,
    SYSTEM_PROMPT,
    OUTPUT_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    PER_DEVICE_BATCH_SIZE,
    MAX_TRAIN_SAMPLES,
    LORA_R,
    LORA_ALPHA,
    DOMAINS,
)
from guardrails import validate_query

# ── Global model cache so we don't reload on every inference call ──────────────
_cached_model = None
_cached_tokenizer = None


def get_inference_model():
    global _cached_model, _cached_tokenizer
    if _cached_model is None:
        from inference import load_model
        _cached_model, _cached_tokenizer = load_model()
    return _cached_model, _cached_tokenizer


def reset_model_cache():
    """Call after training so the next inference loads the freshly trained adapter."""
    global _cached_model, _cached_tokenizer
    _cached_model = None
    _cached_tokenizer = None


# ── GPU info helper ────────────────────────────────────────────────────────────
def get_gpu_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        return f"🟢 **{name}** ({mem} GB VRAM)"
    return "🔴 No GPU detected"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def start_training(domain, lr, epochs, batch_size, max_samples, lora_r, lora_alpha):
    """Called by the Gradio UI. Yields progress strings until training is done."""

    if not torch.cuda.is_available():
        yield "❌ **Error:** No CUDA GPU detected. Training requires a GPU."
        return

    yield f"⏳ Starting **{domain}** training run...\n"
    yield f"  • Learning Rate: `{lr}`\n  • Epochs: `{epochs}`\n  • Batch Size: `{batch_size}`\n  • Max Samples: `{int(max_samples)}`\n  • LoRA R: `{int(lora_r)}`\n  • LoRA Alpha: `{int(lora_alpha)}`\n\n"

    # Monkey-patch config values at runtime so train.py picks them up
    import config as cfg
    cfg.ACTIVE_DOMAIN = domain
    domain_cfg = cfg.DOMAINS[domain]
    cfg.DATASET_NAME = domain_cfg["dataset"]
    cfg.DATA_FILES = domain_cfg["data_files"]
    cfg.SYSTEM_PROMPT = domain_cfg["system_prompt"]
    cfg.OUTPUT_DIR = domain_cfg["output_dir"]
    cfg.DATASET_COLUMNS = domain_cfg["dataset_columns"]
    cfg.LEARNING_RATE = lr
    cfg.NUM_EPOCHS = int(epochs)
    cfg.PER_DEVICE_BATCH_SIZE = int(batch_size)
    cfg.LORA_R = int(lora_r)
    cfg.LORA_ALPHA = int(lora_alpha)

    from train import run_training

    progress_lines = []
    result = {}
    error_holder = []

    def progress_callback(step_logs, summary_line):
        progress_lines.append(summary_line)

    def _train():
        try:
            metrics = run_training(
                lr=lr,
                epochs=int(epochs),
                batch_size=int(batch_size),
                max_samples=int(max_samples),
                progress_callback=progress_callback,
            )
            result.update(metrics)
        except Exception as e:
            error_holder.append(str(e))

    thread = threading.Thread(target=_train)
    thread.start()

    import time
    while thread.is_alive():
        if progress_lines:
            yield "```\n" + "\n".join(progress_lines[-10:]) + "\n```\n"
        time.sleep(3)

    thread.join()

    if error_holder:
        yield f"❌ **Training failed:**\n```\n{error_holder[0]}\n```"
        return

    # Invalidate model cache so next inference loads the fresh adapter
    reset_model_cache()

    final_loss = result.get("train_loss", "N/A")
    if isinstance(final_loss, float):
        final_loss = f"{final_loss:.4f}"

    runtime = result.get("train_runtime", 0)
    summary = f"""
✅ **Training Complete!**

| Metric | Value |
|---|---|
| Final Train Loss | `{final_loss}` |
| Runtime | `{runtime:.1f}s` |
| Adapter saved to | `{result.get("adapter_path", OUTPUT_DIR)}` |

The model cache has been refreshed. You can now test your model in the **🧪 Inference** tab!
"""
    yield "```\n" + "\n".join(progress_lines) + "\n```\n" + summary


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation_ui():
    """Called by the Gradio UI. Returns evaluation metrics as a formatted string."""
    yield "⏳ Loading model and evaluation dataset..."

    try:
        from evaluate import run_evaluation
        metrics = run_evaluation()
    except Exception as e:
        yield f"❌ **Evaluation failed:**\n```\n{e}\n```"
        return

    if "error" in metrics:
        yield f"❌ {metrics['error']}"
        return

    speed_rows = "\n".join(
        f"| Sample {s['sample']} | {s['tokens']} | {s['time_s']}s | {s['tokens_per_sec']} |"
        for s in metrics.get("samples", [])
    )

    report = f"""
✅ **Evaluation Complete!**

---

### 📊 Core Metrics

| Metric | Value | Interpretation |
|---|---|---|
| Validation Loss | `{metrics['val_loss']}` | Lower is better |
| Perplexity | `{metrics['perplexity']}` | Lower is better (1.0 = perfect) |
| Avg Speed | `{metrics['avg_tokens_per_sec']} tok/s` | Higher is better |

---

### ⚡ Generation Speed (per sample)

| Sample | Tokens | Time | Speed |
|---|---|---|---|
{speed_rows}
"""
    yield report


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def chat_response(message, history):
    """Streaming chat function for the ChatInterface tab."""
    if not validate_query(message):
        domain = ACTIVE_DOMAIN
        yield f"🛡️ **[GUARDRAIL BLOCKED]** This question is outside my **{domain}** domain. Please ask a {domain}-related question."
        return

    try:
        from transformers import TextIteratorStreamer
        model, tokenizer = get_inference_model()
    except Exception as e:
        yield f"❌ **Model load failed:** {e}\n\nMake sure you have trained and saved an adapter first."
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
domain_names = list(DOMAINS.keys())

with gr.Blocks(title="QLoRA Fine-Tuning Dashboard", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        f"""
        # 🤖 QLoRA Fine-Tuning Dashboard
        **Active Domain:** `{ACTIVE_DOMAIN.upper()}` &nbsp;|&nbsp; {get_gpu_info()}

        Fine-tune, evaluate, and chat with your domain-specific LoRA adapter — all in one place.
        """
    )

    with gr.Tabs():

        # ── Tab 1: Training ───────────────────────────────────────────────────
        with gr.Tab("🏋️ Training"):
            gr.Markdown("### Configure & Launch Training")
            gr.Markdown(
                "Set your hyperparameters below and click **Start Training**. "
                "Training runs in the background and streams live progress here."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    t_domain = gr.Dropdown(
                        choices=domain_names,
                        value=ACTIVE_DOMAIN,
                        label="Domain",
                        info="The domain to fine-tune the model for."
                    )
                    t_max_samples = gr.Slider(
                        minimum=50,
                        maximum=5000,
                        value=MAX_TRAIN_SAMPLES or 500,
                        step=50,
                        label="Max Training Samples",
                        info="Fewer samples = faster run. Increase for a full training run."
                    )
                    t_epochs = gr.Slider(minimum=1, maximum=10, value=NUM_EPOCHS, step=1, label="Epochs")
                    t_lr = gr.Number(value=LEARNING_RATE, label="Learning Rate", precision=6)
                    t_batch = gr.Slider(minimum=1, maximum=8, value=PER_DEVICE_BATCH_SIZE, step=1, label="Batch Size per Device")
                    t_lora_r = gr.Slider(minimum=4, maximum=64, value=LORA_R, step=4, label="LoRA Rank (r)")
                    t_lora_alpha = gr.Slider(minimum=8, maximum=128, value=LORA_ALPHA, step=8, label="LoRA Alpha")
                    t_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

                with gr.Column(scale=2):
                    t_output = gr.Markdown(
                        label="Training Progress",
                        value="Training output will appear here once you click Start Training."
                    )

            t_btn.click(
                fn=start_training,
                inputs=[t_domain, t_lr, t_epochs, t_batch, t_max_samples, t_lora_r, t_lora_alpha],
                outputs=t_output,
            )

        # ── Tab 2: Evaluation ─────────────────────────────────────────────────
        with gr.Tab("📊 Evaluation"):
            gr.Markdown("### Evaluate the Trained Adapter")
            gr.Markdown(
                "Click **Run Evaluation** to calculate Validation Loss, Perplexity, "
                "and Generation Speed on your validation set."
            )
            e_btn = gr.Button("📊 Run Evaluation", variant="primary", size="lg")
            e_output = gr.Markdown(
                label="Evaluation Results",
                value="Evaluation results will appear here."
            )
            e_btn.click(fn=run_evaluation_ui, inputs=[], outputs=e_output)

        # ── Tab 3: Inference ──────────────────────────────────────────────────
        with gr.Tab("🧪 Inference"):
            gr.Markdown("### Chat with Your Fine-Tuned Model")
            gr.Markdown(
                f"Domain guardrails are **active**. Only `{ACTIVE_DOMAIN}` questions will be answered."
            )

            domain_examples = {
                "medical": [
                    "What are the early warning signs of Type 2 Diabetes?",
                    "What is the difference between Type 1 and Type 2 Diabetes?",
                ],
                "coding": [
                    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.",
                    "Explain the difference between a process and a thread in operating systems.",
                ],
                "law": [
                    "What are the key elements required to form a legally binding contract?",
                    "Explain the difference between civil and criminal law.",
                ],
                "finance": [
                    "What is the difference between a stock and a bond?",
                    "Explain what dollar-cost averaging means and how it works.",
                ],
            }
            examples = domain_examples.get(ACTIVE_DOMAIN, [])

            gr.ChatInterface(
                fn=chat_response,
                examples=examples,
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
