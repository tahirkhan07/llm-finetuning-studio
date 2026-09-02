from typing import Any, Dict
from .detector import HardwareInfo

def recommend_config(hw: HardwareInfo, model_params_billions: float = 7.0) -> Dict[str, Any]:
    """
    Recommends a safe TrainingConfig based on hardware constraints.
    Returns a dictionary of recommended kwargs.
    """
    cfg = {
        "max_seq_length": 1024,
        "learning_rate": 2e-4,
        "num_epochs": 1,
        "precision": "bf16" if hw.bf16_supported else "fp16"
    }
    
    if not hw.cuda_available:
        # Fallback to CPU-friendly setup (though not really recommended for finetuning)
        cfg["method"] = "lora"
        cfg["quantization"] = {"bits": 8, "dtype": "fp32"}
        cfg["per_device_batch_size"] = 1
        cfg["gradient_accumulation_steps"] = 1
        cfg["gradient_checkpointing"] = False
        return cfg

    # If VRAM is tight (e.g., RTX 3060 12GB or 4060 8GB)
    if hw.gpu_vram_gb < 16:
        cfg["method"] = "qlora"
        cfg["quantization"] = {"bits": 4, "dtype": "nf4"}
        cfg["per_device_batch_size"] = 1
        cfg["gradient_accumulation_steps"] = 8
        cfg["gradient_checkpointing"] = True
    # If plenty of VRAM (e.g., RTX 3090/4090 24GB or A100)
    elif hw.gpu_vram_gb >= 24 and model_params_billions <= 7.0:
        cfg["method"] = "lora" # Standard LoRA in FP16/BF16
        cfg["quantization"] = {"bits": 16, "dtype": "fp16"}
        cfg["per_device_batch_size"] = 4
        cfg["gradient_accumulation_steps"] = 2
        cfg["gradient_checkpointing"] = False
    else:
        # Middle ground
        cfg["method"] = "qlora"
        cfg["quantization"] = {"bits": 4, "dtype": "nf4"}
        cfg["per_device_batch_size"] = 2
        cfg["gradient_accumulation_steps"] = 4
        cfg["gradient_checkpointing"] = True
        
    return cfg
