import pytest
from core.hardware.memory import estimate_vram
from core.hardware.detector import HardwareInfo
from core.hardware.planner import recommend_config

def test_estimate_vram():
    # 7B model, seq_len 1024, batch 1, 4bit
    vram, breakdown = estimate_vram(7.0, 1024, 1, is_4bit=True)
    assert breakdown["model_weights"] == 7.0
    assert breakdown["optimizer_state"] == 14.0
    assert breakdown["activations"] == 0.5
    assert vram == 21.5

def test_planner_cpu_fallback():
    hw = HardwareInfo(
        gpu_name="No GPU", gpu_vram_gb=0, gpu_count=0,
        cuda_available=False, cuda_version=None,
        bf16_supported=False, flash_attention2=False,
        cpu_ram_gb=16, free_disk_gb=100
    )
    cfg = recommend_config(hw)
    assert cfg["method"] == "lora"
    assert cfg["precision"] == "fp16"
    assert cfg["quantization"]["bits"] == 8 # CPU fallback default in our planner

def test_planner_low_vram():
    hw = HardwareInfo(
        gpu_name="RTX 4060", gpu_vram_gb=8, gpu_count=1,
        cuda_available=True, cuda_version="11.8",
        bf16_supported=True, flash_attention2=True,
        cpu_ram_gb=32, free_disk_gb=100
    )
    cfg = recommend_config(hw)
    assert cfg["method"] == "qlora"
    assert cfg["quantization"]["bits"] == 4
    assert cfg["gradient_checkpointing"] is True
    assert cfg["precision"] == "bf16"

def test_planner_high_vram():
    hw = HardwareInfo(
        gpu_name="RTX 4090", gpu_vram_gb=24, gpu_count=1,
        cuda_available=True, cuda_version="11.8",
        bf16_supported=True, flash_attention2=True,
        cpu_ram_gb=64, free_disk_gb=100
    )
    cfg = recommend_config(hw, model_params_billions=7.0)
    assert cfg["method"] == "lora"
    assert cfg["quantization"]["bits"] == 16
    assert cfg["gradient_checkpointing"] is False
