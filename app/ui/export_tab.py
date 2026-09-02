"""
Export Tab — Merge adapters into full models, convert to GGUF, and push to HuggingFace Hub.

Provides three sections:
1. Merge a trained LoRA adapter into a full fp16 model
2. Convert a merged model to GGUF format (for Ollama / LM Studio)
3. Push the result to HuggingFace Hub
"""

import os
import threading
import gradio as gr
from core.experiments.tracker import ExperimentTracker
from core.models.exporter import (
    merge_adapter,
    convert_to_gguf,
    push_to_hub,
    list_merged_models,
    list_gguf_files,
    SUPPORTED_GGUF_QUANT_TYPES,
)


def build():
    with gr.Tab("📦 Export"):
        gr.Markdown("### Export Your Fine-Tuned Model")
        gr.Markdown(
            "Merge your trained LoRA adapter into a full model, optionally convert to GGUF "
            "for tools like **Ollama** and **LM Studio**, or push directly to your **HuggingFace** account."
        )

        tracker = ExperimentTracker()

        # ═════════════════════════════════════════════════════════════════
        # SECTION 1: Merge Adapter
        # ═════════════════════════════════════════════════════════════════
        gr.Markdown("---")
        gr.Markdown("#### 1️⃣ Merge LoRA Adapter → Full Model")

        with gr.Row():
            with gr.Column(scale=1):
                merge_adapter_dropdown = gr.Dropdown(
                    label="Select Trained Adapter",
                    choices=[],
                    info="Choose a completed training experiment to merge.",
                )
                merge_refresh_btn = gr.Button("🔄 Refresh Adapters", size="sm")

                merge_base_model = gr.Textbox(
                    label="Base Model ID",
                    placeholder="Auto-detected from adapter config",
                    info="Override the base model ID if needed. Leave blank for auto-detection.",
                )
                merge_output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./outputs/merged",
                    info="Where to save the merged model.",
                )
                merge_btn = gr.Button("🔀 Merge Adapter", variant="primary", size="lg")

            with gr.Column(scale=2):
                merge_output = gr.Markdown(
                    value="Select a trained adapter and click **Merge Adapter** to create a full model.",
                )

        def refresh_adapters():
            """Scan experiments for completed adapters."""
            exps = tracker.list_experiments()
            choices = []
            is_first = True
            for exp in exps:
                if exp["Status"] == "COMPLETED":
                    adapter_path = exp.get("Adapter Path", os.path.join("outputs", exp["Experiment ID"], "adapter"))
                    if os.path.exists(adapter_path):
                        label = f"{exp['Experiment ID']} — {exp['Timestamp']}"
                        if is_first:
                            label += " (Latest)"
                            is_first = False
                        choices.append((label, adapter_path))
            if not choices:
                choices = [("No trained adapters found", "")]
            return gr.update(choices=choices)

        merge_refresh_btn.click(fn=refresh_adapters, outputs=merge_adapter_dropdown)

        def auto_fill_base_model(adapter_path):
            """Read base_model_name_or_path from adapter_config.json."""
            if not adapter_path:
                return ""
            config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                return cfg.get("base_model_name_or_path", "")
            return ""

        merge_adapter_dropdown.change(
            fn=auto_fill_base_model,
            inputs=[merge_adapter_dropdown],
            outputs=[merge_base_model],
        )

        def do_merge(adapter_path, base_model_id, output_dir):
            """Run merge in the current thread, yielding progress updates."""
            if not adapter_path:
                yield "❌ **Error:** Please select a trained adapter first. Click 🔄 Refresh to scan for adapters."
                return

            progress_lines = []

            def on_progress(msg):
                progress_lines.append(msg)

            # Determine output subdirectory
            adapter_name = os.path.basename(os.path.dirname(adapter_path))
            full_output_dir = os.path.join(output_dir, adapter_name)

            yield "⏳ Starting merge...\n"

            # Run merge in a background thread so we can stream progress
            result_holder = {}
            error_holder = []

            def _merge():
                try:
                    result = merge_adapter(
                        adapter_path=adapter_path,
                        output_dir=full_output_dir,
                        base_model_id=base_model_id.strip() or None,
                        progress_fn=on_progress,
                    )
                    result_holder.update(result)
                except Exception as e:
                    error_holder.append(str(e))

            thread = threading.Thread(target=_merge)
            thread.start()

            import time
            while thread.is_alive():
                if progress_lines:
                    yield "\n".join(progress_lines)
                time.sleep(2)

            thread.join()

            if error_holder:
                yield "\n".join(progress_lines) + f"\n\n❌ **Merge failed:**\n```\n{error_holder[0]}\n```"
                return

            final = "\n".join(progress_lines)
            final += f"\n\n| Result | Value |\n|---|---|\n"
            final += f"| Output Directory | `{result_holder.get('output_dir', 'N/A')}` |\n"
            final += f"| Model Size | {result_holder.get('size_gb', 'N/A')} GB |\n"
            final += f"| Shards | {result_holder.get('num_shards', 'N/A')} |\n"
            yield final

        merge_btn.click(
            fn=do_merge,
            inputs=[merge_adapter_dropdown, merge_base_model, merge_output_dir],
            outputs=merge_output,
        )

        # ═════════════════════════════════════════════════════════════════
        # SECTION 2: GGUF Conversion
        # ═════════════════════════════════════════════════════════════════
        gr.Markdown("---")
        gr.Markdown("#### 2️⃣ Convert to GGUF (for Ollama / LM Studio)")

        with gr.Row():
            with gr.Column(scale=1):
                gguf_model_dropdown = gr.Dropdown(
                    label="Merged Model Directory",
                    choices=[],
                    info="Select a merged model to convert.",
                )
                gguf_refresh_btn = gr.Button("🔄 Refresh Merged Models", size="sm")

                gguf_quant = gr.Radio(
                    choices=SUPPORTED_GGUF_QUANT_TYPES,
                    value="Q4_K_M",
                    label="Quantization Type",
                    info="Q4_K_M is a good default (small file, decent quality). F16 keeps full precision.",
                )
                gguf_btn = gr.Button("⚙️ Convert to GGUF", variant="primary", size="lg")

            with gr.Column(scale=2):
                gguf_output = gr.Markdown(
                    value="Merge an adapter first (Section 1), then convert the merged model to GGUF here.",
                )

        def refresh_merged():
            """Scan for merged models."""
            models = list_merged_models()
            if not models:
                return gr.update(choices=["No merged models found — run Merge first"])
            return gr.update(choices=models)

        gguf_refresh_btn.click(fn=refresh_merged, outputs=gguf_model_dropdown)

        def do_gguf(model_dir, quant_type):
            """Run GGUF conversion, yielding progress."""
            if not model_dir or "No merged" in model_dir:
                yield "❌ **Error:** Please merge an adapter first (Section 1), then refresh this list."
                return

            progress_lines = []

            def on_progress(msg):
                progress_lines.append(msg)

            model_name = os.path.basename(model_dir)
            output_path = os.path.join(
                os.path.dirname(model_dir),
                f"{model_name}-{quant_type}.gguf"
            )

            yield "⏳ Starting GGUF conversion...\n"

            result_holder = {}
            error_holder = []

            def _convert():
                try:
                    result = convert_to_gguf(
                        merged_model_dir=model_dir,
                        output_path=output_path,
                        quant_type=quant_type,
                        progress_fn=on_progress,
                    )
                    result_holder.update(result)
                except Exception as e:
                    error_holder.append(str(e))

            thread = threading.Thread(target=_convert)
            thread.start()

            import time
            while thread.is_alive():
                if progress_lines:
                    yield "\n".join(progress_lines)
                time.sleep(2)

            thread.join()

            if error_holder:
                yield "\n".join(progress_lines) + f"\n\n❌ **Conversion failed:**\n```\n{error_holder[0]}\n```"
                return

            final = "\n".join(progress_lines)
            final += f"\n\n| Result | Value |\n|---|---|\n"
            final += f"| Output File | `{result_holder.get('output_path', 'N/A')}` |\n"
            final += f"| File Size | {result_holder.get('size_gb', 'N/A')} GB |\n"
            final += f"| Quantization | {result_holder.get('quant_type', 'N/A')} |\n"
            yield final

        gguf_btn.click(
            fn=do_gguf,
            inputs=[gguf_model_dropdown, gguf_quant],
            outputs=gguf_output,
        )

        # ═════════════════════════════════════════════════════════════════
        # SECTION 3: Push to HuggingFace Hub
        # ═════════════════════════════════════════════════════════════════
        gr.Markdown("---")
        gr.Markdown("#### 3️⃣ Push to HuggingFace Hub")

        with gr.Row():
            with gr.Column(scale=1):
                push_source = gr.Dropdown(
                    label="Model to Upload",
                    choices=[],
                    info="Select a merged model directory to upload.",
                )
                push_refresh_btn = gr.Button("🔄 Refresh Available Models", size="sm")

                push_repo_id = gr.Textbox(
                    label="Repository ID",
                    placeholder="your-username/my-finetuned-model",
                    info="The HuggingFace repo to push to (created automatically if it doesn't exist).",
                )
                push_private = gr.Checkbox(
                    label="Private Repository",
                    value=True,
                    info="If checked, only you can see the model.",
                )
                push_btn = gr.Button("🚀 Push to Hub", variant="primary", size="lg")

            with gr.Column(scale=2):
                push_output = gr.Markdown(
                    value=(
                        "Upload your merged model to HuggingFace Hub to share it or use it in other tools.\n\n"
                        "⚠️ **Requires authentication.** Log in via the **⚙️ Settings** tab first."
                    ),
                )

        def refresh_push_sources():
            """Scan for merged models and GGUF files."""
            sources = list_merged_models()
            if not sources:
                return gr.update(choices=["No models available — merge an adapter first"])
            return gr.update(choices=sources)

        push_refresh_btn.click(fn=refresh_push_sources, outputs=push_source)

        def do_push(model_dir, repo_id, private):
            """Push to Hub, yielding progress."""
            if not model_dir or "No models" in model_dir:
                yield "❌ **Error:** No model to upload. Merge an adapter first."
                return

            if not repo_id or not repo_id.strip():
                yield "❌ **Error:** Please enter a repository ID (e.g. `your-username/my-model`)."
                return

            repo_id = repo_id.strip()
            if "/" not in repo_id:
                yield "❌ **Error:** Repository ID must be in the format `username/model-name`."
                return

            progress_lines = []

            def on_progress(msg):
                progress_lines.append(msg)

            # Try to detect base model ID for model card
            base_model_id = None
            adapter_path = None
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                base_model_id = cfg.get("_name_or_path", None)

            yield "⏳ Starting upload...\n"

            result_holder = {}
            error_holder = []

            def _push():
                try:
                    result = push_to_hub(
                        model_dir=model_dir,
                        repo_id=repo_id,
                        private=private,
                        adapter_path=adapter_path,
                        base_model_id=base_model_id,
                        progress_fn=on_progress,
                    )
                    result_holder.update(result)
                except Exception as e:
                    error_holder.append(str(e))

            thread = threading.Thread(target=_push)
            thread.start()

            import time
            while thread.is_alive():
                if progress_lines:
                    yield "\n".join(progress_lines)
                time.sleep(2)

            thread.join()

            if error_holder:
                yield "\n".join(progress_lines) + f"\n\n❌ **Push failed:**\n```\n{error_holder[0]}\n```"
                return

            final = "\n".join(progress_lines)
            repo_url = result_holder.get("repo_url", "")
            final += f"\n\n| Result | Value |\n|---|---|\n"
            final += f"| Repository | [{repo_id}]({repo_url}) |\n"
            final += f"| Files Uploaded | {result_holder.get('files_uploaded', 'N/A')} |\n"
            yield final

        push_btn.click(
            fn=do_push,
            inputs=[push_source, push_repo_id, push_private],
            outputs=push_output,
        )
