import torch
from typing import Dict, List
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast

def compare_models(base_model: PreTrainedModel, adapter_model: PeftModel, tokenizer: PreTrainedTokenizerFast, prompts: List[str], max_new_tokens: int = 150) -> Dict[str, Dict[str, str]]:
    """
    Runs the same prompts through the base model and the adapter model to show a side-by-side comparison.
    Assumes adapter_model wraps base_model. We can disable the adapter to get base model outputs.
    """
    device = base_model.device
    results = {}
    
    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt
            
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # 1. Base Model Output (disable adapters)
        with adapter_model.disable_adapter():
            with torch.no_grad():
                base_outputs = adapter_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        base_text = tokenizer.decode(base_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 2. Fine-Tuned Model Output (enable adapters)
        with torch.no_grad():
            ft_outputs = adapter_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        ft_text = tokenizer.decode(ft_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        results[prompt] = {
            "base": base_text.strip(),
            "finetuned": ft_text.strip()
        }
        
    return results
