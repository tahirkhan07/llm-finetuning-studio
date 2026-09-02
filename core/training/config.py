from pydantic import BaseModel
from typing import Literal, List, Optional
import torch

class QuantizationConfig(BaseModel):
    bits: Literal[4, 8, 16, 32] = 4
    dtype: Literal["nf4", "fp4", "fp16", "bf16", "fp32"] = "nf4"
    double_quant: bool = True

class LoRAConfig(BaseModel):
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None

class TrainingConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    dataset_id: Optional[str] = None
    method: Literal["qlora", "lora", "sft"] = "qlora"
    quantization: QuantizationConfig = QuantizationConfig()
    lora: LoRAConfig = LoRAConfig()
    learning_rate: float = 2e-4
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    output_dir: str = "./outputs"
    seed: int = 42
    precision: Literal["fp16", "bf16", "fp32"] = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
