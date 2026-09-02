import gradio as gr
from app.state import state
from core.evaluation.metrics import calculate_perplexity, calculate_metrics
from core.datasets.splitter import split_dataset

def build():
    with gr.Tab("5. Evaluation"):
        gr.Markdown("### Model Evaluation")
        
        eval_btn = gr.Button("Run Full Evaluation (Loss, Perplexity, ROUGE-L, BERTScore, F1)", variant="primary")
            
        eval_results = gr.JSON(label="Evaluation Results")
        
        def run_eval():
            if not state.model or not state.canonical_dataset:
                return {"error": "Model and dataset must be loaded first."}
                
            ds = state.canonical_dataset
            if "validation" not in ds:
                ds = split_dataset(ds)
                state.canonical_dataset = ds
                
            val_ds = ds["validation"]
            
            try:
                # Calculate Validation Loss and Perplexity
                ppl, val_loss = calculate_perplexity(state.model, state.tokenizer, val_ds, batch_size=2)
                
                # Calculate ROUGE-L, BERTScore, and Task-specific F1
                gen_metrics = calculate_metrics(state.model, state.tokenizer, val_ds, batch_size=2)
                
                # Combine results
                results = {
                    "Validation Loss": val_loss,
                    "Perplexity": ppl,
                    "ROUGE-L": gen_metrics.get("ROUGE-L", 0.0),
                    "BERTScore": gen_metrics.get("BERTScore", 0.0),
                    "Task-specific Accuracy / F1": gen_metrics.get("Task-specific Accuracy / F1", 0.0)
                }
                
                return results
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"error": str(e)}
                
        eval_btn.click(fn=run_eval, inputs=[], outputs=eval_results)
