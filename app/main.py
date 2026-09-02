import gradio as gr
from app.ui import model_tab, dataset_tab, hardware_tab, training_tab, evaluation_tab, inference_tab, experiments_tab, settings_tab, export_tab

def build_app():
    # A sleek dark theme with rich accent colors
    custom_theme = gr.themes.Monochrome(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    ).set(
        body_background_fill="*neutral_950",
        block_background_fill="*neutral_900",
        block_border_width="1px",
        block_border_color="*neutral_800",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        slider_color="*primary_500"
    )

    with gr.Blocks(title="LLM Fine-Tuning Studio") as demo:
        with gr.Row():
            gr.Markdown("# 🚀 LLM Fine-Tuning Studio")
        
        gr.Markdown("A professional, modular desktop environment for crafting, evaluating, and serving fine-tuned Large Language Models.")
        
        settings_tab.build()
        model_tab.build()
        dataset_tab.build()
        hardware_tab.build()
        training_tab.build()
        evaluation_tab.build()
        inference_tab.build()
        experiments_tab.build()
        export_tab.build()

    return demo, custom_theme

if __name__ == "__main__":
    demo, theme = build_app()
    # share=False ensures local network only. Using queue for generators (streaming chat).
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=theme)
