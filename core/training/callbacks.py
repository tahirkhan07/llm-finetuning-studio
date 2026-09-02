import json
import os
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from typing import Dict, Any

class ProgressCallback(TrainerCallback):
    """
    Writes training progress to a JSON file so that a UI (like Gradio) 
    can constantly poll it and render live metrics.
    """
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history = []
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        if logs:
            step_data = {
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": logs.get("loss", None),
                "learning_rate": logs.get("learning_rate", None)
            }
            self.history.append(step_data)
            
            with open(self.log_path, 'w') as f:
                json.dump(self.history, f, indent=2)

class EarlyStoppingCallback(TrainerCallback):
    """
    Simplified early stopping based on validation loss.
    """
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        val_loss = metrics.get("eval_loss", None)
        if val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    control.should_training_stop = True
