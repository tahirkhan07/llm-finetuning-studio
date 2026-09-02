import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MODEL_NAME,
)


def get_compute_dtype():
    if torch.cuda.is_available():
        # FP16 is a safe starting point for many consumer NVIDIA GPUs.
        return torch.float16
    return torch.float32


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        token=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def load_qlora_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This QLoRA example expects a CUDA GPU. "
            "Use a CUDA-enabled PyTorch installation."
        )

    compute_dtype = get_compute_dtype()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        token=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    return model


def get_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )
