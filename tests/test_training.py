import pytest
from core.training.config import TrainingConfig, QuantizationConfig, LoRAConfig

def test_training_config_defaults():
    cfg = TrainingConfig(model_id="gpt2")
    assert cfg.method == "qlora"
    assert cfg.quantization.bits == 4
    assert cfg.lora.rank == 16
    assert cfg.learning_rate == 2e-4

def test_training_config_validation():
    # pydantic should catch invalid values
    with pytest.raises(ValueError):
        # bits must be a valid literal (4, 8, 16, 32)
        cfg = TrainingConfig(model_id="gpt2", quantization=QuantizationConfig(bits=5))
