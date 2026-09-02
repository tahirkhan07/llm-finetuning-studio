from typing import Tuple, Dict

def estimate_vram(model_params_billions: float, seq_length: int, batch_size: int, is_4bit: bool = True) -> Tuple[float, Dict[str, float]]:
    """
    Estimates VRAM usage for training in GB.
    Returns: (total_vram_gb, breakdown_dict)
    """
    # Base model memory
    if is_4bit:
        bytes_per_param = 1.0 # 4 bits = 0.5 bytes, plus overhead ~1 byte
    else:
        bytes_per_param = 2.0 # FP16/BF16
        
    model_vram_gb = model_params_billions * bytes_per_param
    
    # Optimizer state memory (assuming AdamW)
    # Typically 2x model params for 8-bit optimizer, 4x for 16-bit, 8x for 32-bit.
    # We will assume a simplified 2x model params for PEFT.
    optim_vram_gb = model_params_billions * 2.0
    
    # Activations (highly dependent on seq length and batch size)
    # Rough heuristic: (batch_size * seq_length * hidden_size * num_layers)
    # We use a generic heuristic: 0.5 GB per 1024 tokens at batch 1 for a ~7B model.
    activation_vram_gb = (seq_length / 1024.0) * batch_size * (model_params_billions / 7.0) * 0.5
    
    total_vram_gb = model_vram_gb + optim_vram_gb + activation_vram_gb
    
    breakdown = {
        "model_weights": round(model_vram_gb, 2),
        "optimizer_state": round(optim_vram_gb, 2),
        "activations": round(activation_vram_gb, 2)
    }
    
    return round(total_vram_gb, 2), breakdown
