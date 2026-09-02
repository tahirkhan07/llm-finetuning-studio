import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from typing import Dict, Any

def benchmark_generation(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, prompt: str = "Explain the theory of relativity.", n_samples: int = 3, max_new_tokens: int = 100) -> Dict[str, Any]:
    """
    Benchmarks generation speed in tokens per second.
    """
    model.eval()
    device = model.device
    
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt
        
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Warmup run
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)
        
    timings = []
    total_tokens_generated = 0
    
    for _ in range(n_samples):
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False)
        end_time = time.perf_counter()
        
        # Calculate only newly generated tokens
        new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        total_tokens_generated += new_tokens
        timings.append(end_time - start_time)
        
    avg_time = sum(timings) / n_samples
    avg_tokens_per_sec = max_new_tokens / avg_time
    
    return {
        "avg_tokens_per_sec": round(avg_tokens_per_sec, 2),
        "avg_latency_sec": round(avg_time, 4),
        "total_samples": n_samples,
        "max_new_tokens": max_new_tokens
    }
