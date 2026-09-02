import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
from typing import Tuple, Dict, Any, Optional

def load_model_for_training(model_id: str, quant_cfg: Optional[Dict[str, Any]] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """
    Loads a pretrained model and its tokenizer, optionally with quantization (e.g. 4-bit bitsandbytes).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "token": True,
    }
    
    if quant_cfg:
        bits = quant_cfg.get("bits", 16)
        if bits == 4:
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=quant_cfg.get("dtype", "nf4"),
                bnb_4bit_compute_dtype=compute_dtype
            )
            model_kwargs["quantization_config"] = bnb_config
        elif bits == 8:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config
        elif bits == 16:
            dtype_str = quant_cfg.get("dtype", "fp16")
            model_kwargs["torch_dtype"] = torch.bfloat16 if dtype_str == "bf16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return model, tokenizer
