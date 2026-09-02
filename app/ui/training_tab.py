import gradio as gr
import json
import os
import threading
from app.state import state
from core.training.config import TrainingConfig, QuantizationConfig, LoRAConfig
from core.training.qlora import QLoRAEngine
from core.training.sft import SFTEngine
from core.datasets.splitter import split_dataset

def build():
    with gr.Tab("4. Training"):
        gr.Markdown("### Hyperparameters & Training Loop")
        
        with gr.Row():
            with gr.Column(scale=1):
                method = gr.Radio(["qlora", "sft"], label="Training Method", value="qlora")
                epochs = gr.Slider(1, 10, step=1, value=1, label="Epochs")
                max_samples = gr.Slider(0, 50000, step=50, value=100, label="Max Training Samples (0 = all)")
                batch_size = gr.Slider(1, 16, step=1, value=2, label="Batch Size (per device)")
                grad_acc = gr.Slider(1, 32, step=1, value=4, label="Gradient Accumulation Steps")
                lr = gr.Number(value=2e-4, label="Learning Rate")
                lora_rank = gr.Slider(4, 128, step=4, value=16, label="LoRA Rank (r)")
                
                train_btn = gr.Button("🚀 Start Training", variant="primary")
                
            with gr.Column(scale=2):
                log_box = gr.Markdown(label="Live Metrics Stream", value="*Metrics will appear here during training...*")
                status_txt = gr.Markdown("Ready.")
                
        def train_thread(method_val, ep, max_s, bs, ga, lr_val, rank):
            if not state.model or not state.canonical_dataset:
                return
                
            # Split dataset if not split
            ds = state.canonical_dataset
            if "train" not in ds:
                ds = split_dataset(ds)
                
            if max_s > 0:
                train_len = min(int(max_s), len(ds["train"]))
                ds["train"] = ds["train"].select(range(train_len))
                if "validation" in ds:
                    val_len = min(int(max_s * 0.1) or 1, len(ds["validation"]))
                    ds["validation"] = ds["validation"].select(range(val_len))
                    
            state.canonical_dataset = ds
                
            cfg = TrainingConfig(
                model_id=state.model_id,
                method=method_val,
                num_epochs=ep,
                per_device_batch_size=bs,
                gradient_accumulation_steps=ga,
                learning_rate=lr_val,
                lora=LoRAConfig(
                    rank=rank,
                    target_modules=getattr(state, 'lora_targets', None),
                ),
                quantization=QuantizationConfig(bits=4) if method_val == "qlora" else QuantizationConfig(bits=16)
            )
            state.training_cfg = cfg
            
            engine_cls = QLoRAEngine if method_val == "qlora" else SFTEngine
            engine = engine_cls(cfg, state.model, state.tokenizer, ds)
            
            state._current_log_path = engine.log_path
            
            try:
                engine.prepare()
                engine.train()
                engine.save()
                
                # Append success message to the log file so the UI can pick it up
                try:
                    with open(engine.log_path, 'r') as f:
                        history = json.load(f)
                except Exception:
                    history = []
                history.append({"status": "✅ Training Complete! Model saved to outputs/."})
                with open(engine.log_path, 'w') as f:
                    json.dump(history, f, indent=2)
                    
            except Exception as e:
                with open(engine.log_path, 'w') as f:
                    json.dump([{"error": str(e)}], f)

        def start_training(method_val, ep, max_s, bs, ga, lr_val, rank):
            if not state.model:
                return "Error: No model loaded in Tab 1.", gr.update()
            if not state.canonical_dataset:
                return "Error: No dataset mapped in Tab 2.", gr.update()
                
            t = threading.Thread(target=train_thread, args=(method_val, ep, max_s, bs, ga, lr_val, rank))
            t.start()
            return "⏳ Training Started in Background...", gr.update()
            
        def poll_logs():
            if not hasattr(state, '_current_log_path') or not os.path.exists(state._current_log_path):
                return gr.update(), gr.update()
            try:
                with open(state._current_log_path, 'r') as f:
                    data = json.load(f)
                    if not data:
                        return gr.update(), "*Waiting for first log entry...*"
                    
                    latest = data[-1]
                    if "status" in latest:
                        return latest["status"], gr.update()
                    if "error" in latest:
                        return f"❌ **Error:** {latest['error']}", gr.update()
                        
                    # Build a Markdown table from the last 10 entries
                    md = "### 📊 Live Training Metrics\n\n"
                    md += "| Epoch | Loss | Learning Rate | Other |\n|---|---|---|---|\n"
                    
                    for entry in reversed(data[-10:]):  # newest on top
                        if "loss" in entry or "train_loss" in entry:
                            step = round(entry.get("epoch", 0), 2)
                            loss_val = entry.get("loss", entry.get("train_loss", "N/A"))
                            if isinstance(loss_val, (float, int)):
                                loss_val = round(loss_val, 4)
                            
                            lr_val = entry.get("learning_rate", "N/A")
                            if isinstance(lr_val, (float, int)):
                                lr_val = f"{lr_val:.2e}"
                                
                            other = ", ".join([f"{k}: {v}" for k, v in entry.items() if k not in ["loss", "train_loss", "epoch", "learning_rate", "step"]])
                            md += f"| {step} | **{loss_val}** | `{lr_val}` | {other} |\n"
                            
                    return "⏳ Training in progress...", md
            except:
                return gr.update(), gr.update()

        train_btn.click(
            fn=start_training,
            inputs=[method, epochs, max_samples, batch_size, grad_acc, lr, lora_rank],
            outputs=[status_txt, log_box]
        )
        
        # Gradio Timer for live polling
        timer = gr.Timer(2.0)
        timer.tick(fn=poll_logs, outputs=[status_txt, log_box])
