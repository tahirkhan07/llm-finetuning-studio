from transformers import PreTrainedModel, PreTrainedTokenizerFast
from typing import Dict, Any

def inspect_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast) -> Dict[str, Any]:
    """
    Inspects model and tokenizer to return a dictionary of properties.
    """
    cfg = model.config
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    has_chat_template = False
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        has_chat_template = True
        
    return {
        "architecture": model.__class__.__name__,
        "model_type": getattr(cfg, "model_type", "unknown"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "hidden_size": getattr(cfg, "hidden_size", "unknown"),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", "unknown"),
        "num_attention_heads": getattr(cfg, "num_attention_heads", "unknown"),
        "has_chat_template": has_chat_template,
        "vocab_size": getattr(cfg, "vocab_size", "unknown"),
    }
