import pytest
from core.models.architecture import get_lora_targets

class DummyModel:
    pass

class LlamaForCausalLM(DummyModel):
    pass

class Qwen2ForCausalLM(DummyModel):
    pass

class UnknownModel(DummyModel):
    def named_modules(self):
        # No nn.Linear layers → triggers fallback
        return iter([])

def test_get_lora_targets_llama():
    model = LlamaForCausalLM()
    targets = get_lora_targets(model)
    assert "q_proj" in targets
    assert "v_proj" in targets
    assert "gate_proj" in targets

def test_get_lora_targets_unknown():
    model = UnknownModel()
    targets = get_lora_targets(model)
    assert targets == ["q_proj", "v_proj"] # Default fallback
