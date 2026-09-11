from fastapi import APIRouter, HTTPException
import asyncio
from app.state import state
from core.evaluation.metrics import calculate_perplexity, calculate_metrics
from core.inference.loader import InferenceLoader
from core.experiments.tracker import ExperimentTracker

router = APIRouter()

@router.post("/run")
async def run_eval():
    if not state.model or not state.canonical_dataset:
        raise HTTPException(status_code=400, detail="Model and dataset must be loaded first.")
        
    ds = state.canonical_dataset
    if "test" in ds:
        eval_ds = ds["test"]
    elif "validation" in ds:
        eval_ds = ds["validation"]
    else:
        eval_ds = ds["train"]
        
    tracker = ExperimentTracker()
    
    try:
        if not hasattr(state.model, 'disable_adapter'):
            exps = tracker.list_experiments()
            completed_exps = [e for e in exps if e["Status"] == "COMPLETED"]
            valid_exps = [
                e for e in completed_exps 
                if not e.get("Config") or e.get("Config", {}).get("model_id") == state.model_id
            ]
            if valid_exps:
                latest_exp = valid_exps[0]
                adapter_path = latest_exp.get("Adapter Path", f"outputs/{latest_exp['Experiment ID']}/adapter")
                
                state.clear_vram()
                m, t = await asyncio.to_thread(InferenceLoader.load, state.model_id, adapter_path)
                state.model = m
                state.tokenizer = t
        
        try:
            ft_ppl, ft_val_loss = await asyncio.to_thread(calculate_perplexity, state.model, state.tokenizer, eval_ds, batch_size=2)
            ft_metrics = await asyncio.to_thread(calculate_metrics, state.model, state.tokenizer, eval_ds, batch_size=2)
            
            base_metrics = None
            if adapter_path:
                # unload adapter to get base model metrics
                # For simplicity in testing, we just reload the base model
                m, t = await asyncio.to_thread(InferenceLoader.load, state.model_id, None)
                state.model = m
                
                base_ppl, base_val_loss = await asyncio.to_thread(calculate_perplexity, state.model, state.tokenizer, eval_ds, batch_size=2)
                base_metrics = await asyncio.to_thread(calculate_metrics, state.model, state.tokenizer, eval_ds, batch_size=2)
            else:
                base_ppl, base_val_loss = ft_ppl, ft_val_loss
                base_metrics = ft_metrics
                ft_ppl, ft_val_loss = "N/A", "N/A"
                ft_metrics = {}
        
        except Exception as e:
            return {"error": f"Failed during evaluation: {str(e)}"}
        
        return {
            "success": True,
            "results": [
                {"metric": "Perplexity", "base": str(base_ppl), "finetuned": str(ft_ppl)},
                {"metric": "ROUGE-L", "base": str(base_metrics.get("ROUGE-L", 0.0)), "finetuned": str(ft_metrics.get("ROUGE-L", 0.0))},
                {"metric": "BLEU", "base": str(base_metrics.get("BLEU", 0.0)), "finetuned": str(ft_metrics.get("BLEU", 0.0))},
                {"metric": "Response latency (s)", "base": str(base_metrics.get("Response latency (s)", 0.0)), "finetuned": str(ft_metrics.get("Response latency (s)", 0.0))},
                {"metric": "Task accuracy (F1)", "base": str(base_metrics.get("Task-specific Accuracy / F1", 0.0)), "finetuned": str(ft_metrics.get("Task-specific Accuracy / F1", 0.0))},
                {"metric": "LLM Correctness (out of 5)", "base": str(base_metrics.get("LLM Correctness (out of 5)", 0.0)), "finetuned": str(ft_metrics.get("LLM Correctness (out of 5)", 0.0))},
                {"metric": "LLM Relevance (out of 5)", "base": str(base_metrics.get("LLM Relevance (out of 5)", 0.0)), "finetuned": str(ft_metrics.get("LLM Relevance (out of 5)", 0.0))},
                {"metric": "LLM Completeness (out of 5)", "base": str(base_metrics.get("LLM Completeness (out of 5)", 0.0)), "finetuned": str(ft_metrics.get("LLM Completeness (out of 5)", 0.0))}
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
