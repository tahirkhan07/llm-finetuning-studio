import gradio as gr
from app.state import state
from core.hardware.detector import detect_hardware
from core.hardware.planner import recommend_config
import json

def build():
    with gr.Tab("3. Hardware"):
        gr.Markdown("### Hardware Diagnostics & Profiling")
        
        with gr.Row():
            with gr.Column():
                detect_btn = gr.Button("Run Diagnostics", variant="primary")
                hw_json = gr.JSON(label="Detected Hardware")
                
            with gr.Column():
                rec_btn = gr.Button("Get Recommended Config", interactive=False)
                rec_json = gr.JSON(label="Recommended Training Profile")
                
        def run_detect():
            hw = detect_hardware()
            state.hardware = hw
            return hw.__dict__, gr.update(interactive=True)
            
        def run_rec():
            if not state.hardware:
                return {"error": "Run diagnostics first"}
            
            # Estimate params if model loaded, else assume 7B
            params = 7.0
            if state.model:
                params = sum(p.numel() for p in state.model.parameters()) / 1e9
                
            cfg = recommend_config(state.hardware, model_params_billions=params)
            return cfg

        detect_btn.click(fn=run_detect, outputs=[hw_json, rec_btn])
        rec_btn.click(fn=run_rec, outputs=[rec_json])
