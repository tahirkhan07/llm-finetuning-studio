import os
from pathlib import Path
from typing import List
from transformers import PreTrainedModel
from peft import PeftModel

def save_adapter(model: PeftModel, output_dir: str) -> None:
    """
    Saves the PEFT adapter weights and config.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

def load_adapter(base_model: PreTrainedModel, adapter_path: str) -> PeftModel:
    """
    Loads a PEFT adapter on top of a base model.
    """
    return PeftModel.from_pretrained(base_model, adapter_path)

def list_adapters(root_dir: str = "./outputs") -> List[Path]:
    """
    Lists all saved adapters in the root directory.
    """
    adapters = []
    root_path = Path(root_dir)
    if not root_path.exists():
        return adapters
        
    for d in root_path.iterdir():
        if d.is_dir() and (d / "adapter_config.json").exists():
            adapters.append(d)
            
    return adapters
