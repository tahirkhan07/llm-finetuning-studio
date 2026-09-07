from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import threading
import json
import os
from app.state import state
from core.training.config import TrainingConfig, QuantizationConfig, LoRAConfig
from core.training.qlora import QLoRAEngine
from core.training.sft import SFTEngine
from core.datasets.splitter import split_dataset

router = APIRouter()

class TrainRequest(BaseModel):
    method: str
    epochs: int
    max_samples: int
    batch_size: int
    grad_acc: int
    learning_rate: float
    lora_rank: int
    max_seq_length: int = 1024
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    eval_steps: int = 10
    save_steps: int = 10
    early_stopping: bool = True
    early_stopping_patience: int = 3

def train_thread(method_val, ep, max_s, bs, ga, lr_val, rank, msl, train_ratio, val_ratio, test_ratio, eval_steps, save_steps, early_stopping, early_stopping_patience):
    import traceback
    
    # Fallback log path in case engine never gets created
    fallback_log_dir = os.path.join("experiments", "latest_run")
    fallback_log_path = os.path.join(fallback_log_dir, "training_log.json")
    os.makedirs(fallback_log_dir, exist_ok=True)
    state._current_log_path = fallback_log_path
    
    # Write an initial status so the UI knows training has started
    with open(fallback_log_path, 'w') as f:
        json.dump([{"status": "Preparing dataset and initializing trainer..."}], f)
    
    engine = None
    
    try:
        if not state.model or not state.canonical_dataset:
            raise Exception("Model or dataset not loaded. Please load both before training.")
            
        ds = state.canonical_dataset
        
        # Recombine if already split so we can resplit dynamically
        if "train" in ds:
            from datasets import concatenate_datasets
            datasets_to_concat = [ds[split] for split in ds.keys()]
            ds = concatenate_datasets(datasets_to_concat)
            
        from core.datasets.splitter import deduplicate_dataset, split_dataset
        ds = deduplicate_dataset(ds)
        ds = split_dataset(ds, train_ratio=train_ratio, val_ratio=val_ratio)
        
        # Save back to state so the Evaluation tab can use the test split later
        state.canonical_dataset = ds
        
        # Only limit the training split!
        if max_s > 0:
            train_len = min(int(max_s), len(ds["train"]))
            ds["train"] = ds["train"].select(range(train_len))
        
        # Print dataset split info to terminal
        print(f"\n{'='*60}")
        print(f"Dataset Splits:")
        for split_name in ds:
            print(f"  {split_name}: {len(ds[split_name])} samples")
        print(f"{'='*60}\n")
            
        # Validation size check
        if "validation" in ds and len(ds["validation"]) < 10:
            raise Exception(f"Validation set contains only {len(ds['validation'])} samples. This is too small for meaningful evaluation. Please increase your validation split or use a larger dataset.")
            
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
            max_seq_length=msl,
            quantization=QuantizationConfig(bits=4) if method_val == "qlora" else QuantizationConfig(bits=16),
            eval_steps=eval_steps,
            save_steps=save_steps,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience
        )
        state.training_cfg = cfg
        
        engine_cls = QLoRAEngine if method_val == "qlora" else SFTEngine
        engine = engine_cls(cfg, state.model, state.tokenizer, ds)
        
        # Now that engine exists, update the log path to the real one
        state._current_log_path = engine.log_path
        
        # Write initial status to the real log path
        with open(engine.log_path, 'w') as f:
            json.dump([{"status": "Trainer initialized. Starting training loop..."}], f)
            
        engine.prepare()
        engine.train()
        engine.save()
        
        try:
            with open(engine.log_path, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
        history.append({"status": "Training Complete! Model saved to outputs/."})
        with open(engine.log_path, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ TRAINING ERROR: {error_msg}")
        traceback.print_exc()
        
        # Write error to whichever log path is active
        log_path = engine.log_path if engine else state._current_log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump([{"error": error_msg}], f)


@router.post("/start")
async def start_training(req: TrainRequest):
    if not state.model:
        raise HTTPException(status_code=400, detail="No model loaded in Tab 1.")
    if not state.canonical_dataset:
        raise HTTPException(status_code=400, detail="No dataset mapped in Tab 2.")
        
    t = threading.Thread(target=train_thread, args=(
        req.method, req.epochs, req.max_samples, req.batch_size, 
        req.grad_acc, req.learning_rate, req.lora_rank, req.max_seq_length,
        req.train_ratio, req.val_ratio, req.test_ratio,
        req.eval_steps, req.save_steps, req.early_stopping, req.early_stopping_patience
    ))
    t.start()
    return {"success": True, "message": "Training Started in Background..."}

@router.get("/logs")
async def poll_logs():
    if not hasattr(state, '_current_log_path') or not os.path.exists(state._current_log_path):
        return {"logs": []}
    try:
        with open(state._current_log_path, 'r') as f:
            data = json.load(f)
            return {"logs": data}
    except Exception:
        return {"logs": []}
