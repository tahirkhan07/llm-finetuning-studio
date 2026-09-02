from typing import List
from transformers import PreTrainedModel

# Maps model architecture class names to the recommended LoRA target modules.
# This registry is a performance optimization — if a model's architecture is
# not listed here, we auto-detect all linear layers at runtime.
ARCHITECTURE_REGISTRY = {
    "LlamaForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "MistralForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Qwen2ForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "GemmaForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Gemma2ForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "Gemma3ForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "PhiForCausalLM": ["q_proj", "k_proj", "v_proj", "dense"],
    "Phi3ForCausalLM": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "MixtralForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
    "StableLmForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _auto_detect_linear_targets(model: PreTrainedModel) -> List[str]:
    """
    Scans the model for all unique nn.Linear layer names (excluding lm_head).
    This allows us to support ANY model architecture, even ones not in our registry.
    """
    import torch.nn as nn

    target_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get the last part of the dotted name (e.g. "layers.0.self_attn.q_proj" -> "q_proj")
            short_name = name.split(".")[-1]
            if short_name != "lm_head":
                target_names.add(short_name)
    return sorted(target_names)


def get_lora_targets(model: PreTrainedModel) -> List[str]:
    """
    Returns the recommended list of LoRA target modules based on the model's architecture.
    If the architecture is unknown, auto-detects all linear layers in the model.
    """
    arch = model.__class__.__name__
    targets = ARCHITECTURE_REGISTRY.get(arch)

    if targets is not None:
        return targets

    # Auto-detect for unknown architectures
    detected = _auto_detect_linear_targets(model)
    return detected if detected else ["q_proj", "v_proj"]
