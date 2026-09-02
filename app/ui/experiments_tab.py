import gradio as gr
import pandas as pd
from core.experiments.tracker import ExperimentTracker

def build():
    with gr.Tab("7. Experiments"):
        gr.Markdown("### Experiment Tracking Database")
        
        tracker = ExperimentTracker()
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Database", variant="primary")
            
        df_display = gr.Dataframe(headers=["Experiment ID", "Status", "Final Loss", "Timestamp", "Path"], interactive=False)
        
        def load_data():
            data = tracker.list_experiments()
            if not data:
                return pd.DataFrame(columns=["Experiment ID", "Status", "Final Loss", "Timestamp", "Path"])
            return pd.DataFrame(data)
            
        refresh_btn.click(fn=load_data, outputs=df_display)
