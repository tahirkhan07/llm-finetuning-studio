import gradio as gr
from app.state import state
from core.inference.loader import InferenceLoader
from core.inference.generator import Generator
from core.inference.guardrails import Guardrails
from core.experiments.tracker import ExperimentTracker

def build():
    with gr.Tab("6. Inference"):
        gr.Markdown("### Interactive Chat & Inference")
        
        with gr.Row():
            with gr.Column(scale=1):
                adapter_dropdown = gr.Dropdown(label="Select Trained Adapter", choices=["None"])
                refresh_btn = gr.Button("🔄 Refresh List")
                load_btn = gr.Button("Load Adapter", variant="primary")
                
                gr.Markdown("---")
                guardrails_toggle = gr.Checkbox(label="🛡️ Enable Safety Guardrails", value=True)
                temp = gr.Slider(0.0, 2.0, step=0.1, value=0.7, label="Temperature")
                max_toks = gr.Slider(10, 1024, step=10, value=256, label="Max New Tokens")
                
                status = gr.Markdown("*Ready.*")
                
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(placeholder="Type a message...", show_label=False)
                clear = gr.Button("Clear Chat")
                
        # Initialize components lazily
        tracker = ExperimentTracker()
        
        def refresh_adapters():
            exps = tracker.list_experiments()
            choices = ["None"]
            is_first = True
            for e in exps:
                if e["Status"] == "COMPLETED":
                    adapter_path = e.get("Adapter Path", f"outputs/{e['Experiment ID']}/adapter")
                    label = f"{e['Experiment ID']} — {e['Timestamp']}"
                    if is_first:
                        label += " (Latest)"
                        is_first = False
                    choices.append((label, adapter_path))
            return gr.update(choices=choices)
            
        refresh_btn.click(fn=refresh_adapters, outputs=adapter_dropdown)
        
        def load_adapter_for_chat(adapter_path):
            if not state.model_id:
                return "Error: Base model ID not set. Go to Tab 1."
            
            try:
                state.clear_vram()
                path = None if adapter_path == "None" else adapter_path
                m, t = InferenceLoader.load(state.model_id, path)
                state.inference_model = m
                state.inference_tokenizer = t
                state.inference_adapter_path = path
                return f"✅ Loaded adapter: {adapter_path}"
            except Exception as e:
                return f"❌ Error: {str(e)}"
                
        load_btn.click(fn=load_adapter_for_chat, inputs=adapter_dropdown, outputs=status)
        
        guard = Guardrails()
        
        def bot_stream(user_message, history, use_guard, t, m):
            if not state.inference_model:
                yield history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "Error: No model loaded for inference. Click 'Load Adapter'."}]
                return
                
            if use_guard and not guard.validate(user_message):
                yield history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "🛡️ **Guardrails Triggered:** I cannot fulfill this request."}]
                return
                
            # Build conversation safely for Gradio 6 (ChatMessage objects or dicts)
            messages = []
            for h in history:
                if isinstance(h, dict):
                    role = h.get("role", "user")
                    content = h.get("content", "")
                elif hasattr(h, "role") and hasattr(h, "content"):
                    role = getattr(h, "role")
                    content = getattr(h, "content")
                elif isinstance(h, (list, tuple)) and len(h) >= 2:
                    messages.append({"role": "user", "content": str(h[0])})
                    messages.append({"role": "assistant", "content": str(h[1])})
                    continue
                else:
                    continue
                    
                # In case Gradio passes multimedia as a list, extract the string
                if isinstance(content, (list, tuple)):
                    content = str(content[0]) if len(content) > 0 else ""
                elif not isinstance(content, str):
                    content = str(content)
                    
                messages.append({"role": role, "content": content})
                
            messages.append({"role": "user", "content": str(user_message)})
            
            gen = Generator(state.inference_model, state.inference_tokenizer)
            stream = gen.generate_stream(messages, max_new_tokens=m, temperature=t)
            
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": ""})
            for chunk in stream:
                history[-1]["content"] += chunk
                yield history

        msg.submit(bot_stream, inputs=[msg, chatbot, guardrails_toggle, temp, max_toks], outputs=[chatbot]).then(
            lambda: "", None, msg
        )
        clear.click(lambda: None, None, chatbot, queue=False)
