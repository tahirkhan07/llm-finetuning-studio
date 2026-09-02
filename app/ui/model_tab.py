import gradio as gr
import pandas as pd
from app.state import state
from core.models.loader import load_model_for_training
from core.models.inspector import inspect_model
from core.models.architecture import get_lora_targets

def build():
    with gr.Tab("1. Model"):
        gr.Markdown("### Select & Inspect Base Model")
        gr.Markdown(
            "Pick a model from the dropdown **or paste any HuggingFace model ID** "
            "(e.g. `NousResearch/Hermes-2-Pro-Llama-3-8B`). "
            "Gated models (🔒 Gemma, Llama) require `huggingface-cli login` first."
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                from core.models.registry import POPULAR_MODELS
                model_id_input = gr.Dropdown(
                    choices=POPULAR_MODELS,
                    label="HuggingFace Model ID or Local Path",
                    value="Qwen/Qwen2.5-0.5B-Instruct",
                    allow_custom_value=True,
                    info="Type or paste any model ID — not limited to this list."
                )
                quant_dropdown = gr.Dropdown(
                    choices=["4-bit (QLoRA)", "8-bit", "16-bit (LoRA)"],
                    value="4-bit (QLoRA)",
                    label="Load Precision"
                )
                load_btn = gr.Button("Load & Inspect Model", variant="primary")
            
            with gr.Column(scale=3):
                status_box = gr.Markdown("*No model loaded yet.*")
                arch_table = gr.Dataframe(headers=["Property", "Value"], interactive=False, visible=False)
                template_info = gr.Markdown(visible=False)

        def load_and_inspect(model_id, quant):
            try:
                # Map UI precision to config
                quant_cfg = {"bits": 4, "dtype": "nf4"}
                if quant == "8-bit":
                    quant_cfg = {"bits": 8}
                elif quant == "16-bit (LoRA)":
                    quant_cfg = {"bits": 16, "dtype": "bf16"}
                    
                state.clear_vram()
                model, tokenizer = load_model_for_training(model_id, quant_cfg)
                state.model_id = model_id
                state.model = model
                state.tokenizer = tokenizer
                
                props = inspect_model(model, tokenizer)
                df = pd.DataFrame(list(props.items()), columns=["Property", "Value"])
                
                # Auto-detect LoRA targets for this architecture
                lora_targets = get_lora_targets(model)
                state.lora_targets = lora_targets
                
                # Build chat template status message
                has_template = props.get("has_chat_template", False)
                if has_template:
                    tmpl_msg = (
                        "✅ **Chat template detected.** This model's tokenizer has a built-in chat template. "
                        "During training, your dataset's `messages` will be automatically formatted "
                        "using this template (e.g. Llama format, Gemma format, ChatML, etc.).\n\n"
                        f"**Auto-detected LoRA targets:** `{', '.join(lora_targets)}`"
                    )
                else:
                    tmpl_msg = (
                        "⚠️ **No chat template found.** This model's tokenizer does not define a chat template. "
                        "A default ChatML template will be applied. For best results, use an `-Instruct` or `-Chat` variant.\n\n"
                        f"**Auto-detected LoRA targets:** `{', '.join(lora_targets)}`"
                    )
                
                return (
                    f"✅ **Successfully loaded:** `{model_id}`",
                    gr.update(value=df, visible=True),
                    gr.update(value=tmpl_msg, visible=True),
                )
            except Exception as e:
                return (
                    f"❌ **Error:** {str(e)}",
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

        load_btn.click(
            fn=load_and_inspect,
            inputs=[model_id_input, quant_dropdown],
            outputs=[status_box, arch_table, template_info]
        )
