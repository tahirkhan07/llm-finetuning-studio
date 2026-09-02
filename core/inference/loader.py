import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast
from peft import PeftModel
from typing import Tuple, Optional

class InferenceLoader:
    @staticmethod
    def load(base_model_id: str, adapter_path: Optional[str] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        """
        Loads the base model and optionally applies a trained adapter for inference.
        Uses 16-bit precision by default for faster inference.
        """
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, token=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=True,
        )
        
        if adapter_path:
            # Wrap with adapter
            model = PeftModel.from_pretrained(model, adapter_path)
            
        model.eval()
        return model, tokenizer
