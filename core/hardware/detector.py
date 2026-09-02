from dataclasses import dataclass
import torch
import GPUtil
import psutil
import shutil
import os
from typing import Optional

@dataclass
class HardwareInfo:
    gpu_name: str
    gpu_vram_gb: float
    gpu_count: int
    cuda_available: bool
    cuda_version: Optional[str]
    bf16_supported: bool
    flash_attention2: bool
    cpu_ram_gb: float
    free_disk_gb: float

def detect_hardware() -> HardwareInfo:
    """Detects available hardware capabilities and returns a summary."""
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None
    
    gpus = GPUtil.getGPUs()
    
    if cuda_available and gpus:
        gpu = gpus[0]
        gpu_name = gpu.name
        gpu_vram_gb = gpu.memoryTotal / 1024.0 # Convert MB to GB
        gpu_count = len(gpus)
        
        # Check BF16 support (Ampere or newer, compute capability >= 8.0)
        try:
            bf16_supported = torch.cuda.get_device_capability(gpu.id)[0] >= 8
        except Exception:
            bf16_supported = False
    else:
        gpu_name = "No GPU detected"
        gpu_vram_gb = 0.0
        gpu_count = 0
        bf16_supported = False

    # Check Flash Attention 2
    try:
        import flash_attn
        flash_attention2 = True
    except ImportError:
        flash_attention2 = False

    # System memory and disk
    ram_gb = psutil.virtual_memory().total / (1024**3)
    free_disk_gb = shutil.disk_usage("/").free / (1024**3)

    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_count=gpu_count,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        bf16_supported=bf16_supported,
        flash_attention2=flash_attention2,
        cpu_ram_gb=ram_gb,
        free_disk_gb=free_disk_gb
    )
