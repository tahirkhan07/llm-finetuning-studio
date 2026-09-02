import time
import math

import torch
from tqdm import tqdm

from config import ACTIVE_DOMAIN
from dataset import load_domain_dataset
from inference import load_model


def benchmark_generation(model, tokenizer, eval_dataset, num_samples=5):
    """
    Measures generation speed (tokens/sec) on a subset of the eval data.
    Returns a dict with per-sample results and average speed.
    """
    results = []
    total_time = 0
    total_new_tokens = 0

    for i in range(min(num_samples, len(eval_dataset))):
        messages = eval_dataset[i]["messages"][:-1]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
            )
        duration = time.time() - start_time

        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        tps = new_tokens / duration if duration > 0 else 0

        results.append({
            "sample": i + 1,
            "tokens": new_tokens,
            "time_s": round(duration, 2),
            "tokens_per_sec": round(tps, 2),
        })

        total_time += duration
        total_new_tokens += new_tokens

    avg_tps = round(total_new_tokens / total_time, 2) if total_time > 0 else 0
    return {"samples": results, "avg_tokens_per_sec": avg_tps}


def calculate_perplexity(model, tokenizer, eval_dataset, progress_callback=None):
    """
    Calculates the validation loss and perplexity across the dataset.
    Returns a dict with loss and perplexity values.
    """
    total_loss = 0
    total_length = 0

    for i in tqdm(range(len(eval_dataset)), desc="Calculating perplexity"):
        messages = eval_dataset[i]["messages"]

        encodings = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        seq_len = encodings["input_ids"].size(1)

        with torch.no_grad():
            outputs = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                labels=encodings["input_ids"],
            )
            total_loss += outputs.loss.item() * seq_len
            total_length += seq_len

        if progress_callback:
            progress_callback(i + 1, len(eval_dataset))

    avg_loss = total_loss / total_length
    perplexity = math.exp(avg_loss)

    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
    }


def run_evaluation(num_speed_samples=5, progress_callback=None):
    """
    Main evaluation entry point. Returns all metrics in a single dict.
    """
    model, tokenizer = load_model()
    _, eval_dataset = load_domain_dataset()

    if len(eval_dataset) == 0:
        return {"error": "No evaluation data found!"}

    speed_metrics = benchmark_generation(model, tokenizer, eval_dataset, num_samples=num_speed_samples)
    perplexity_metrics = calculate_perplexity(model, tokenizer, eval_dataset, progress_callback=progress_callback)

    return {**speed_metrics, **perplexity_metrics}


def main():
    print("=" * 70)
    print(f"Evaluating {ACTIVE_DOMAIN.capitalize()} QLoRA Model")
    print("=" * 70)

    metrics = run_evaluation()

    print(f"\n--- Generation Speed ---")
    for s in metrics["samples"]:
        print(f"  Sample {s['sample']}: {s['tokens']} tokens in {s['time_s']}s ({s['tokens_per_sec']} tok/s)")
    print(f"  Average: {metrics['avg_tokens_per_sec']} tokens/sec")

    print(f"\n--- Perplexity ---")
    print(f"  Validation Loss:  {metrics['val_loss']}")
    print(f"  Perplexity:       {metrics['perplexity']}")


if __name__ == "__main__":
    main()
