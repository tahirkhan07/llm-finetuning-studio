import os
import torch
from trl import SFTConfig, SFTTrainer

from config import (
    ACTIVE_DOMAIN,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    OUTPUT_DIR,
    PER_DEVICE_BATCH_SIZE,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
)
from dataset import load_domain_dataset
from model import get_lora_config, load_qlora_model, load_tokenizer


def run_training(
    lr: float = LEARNING_RATE,
    epochs: int = NUM_EPOCHS,
    batch_size: int = PER_DEVICE_BATCH_SIZE,
    max_samples: int = None,
    progress_callback=None,
):
    """
    Core training routine. Can be called from the CLI or the Gradio UI.
    Returns a dict with final training metrics.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    logs = []
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    logs.append(f"GPU: {gpu_name} ({gpu_mem} GB)")

    tokenizer = load_tokenizer()
    model = load_qlora_model()

    train_dataset, eval_dataset = load_domain_dataset(max_samples=max_samples)
    logs.append(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)} examples")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field=None,
        max_seq_length=MAX_SEQ_LENGTH,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
        remove_unused_columns=True,
    )

    peft_config = get_lora_config()

    # Custom callback to stream progress
    step_logs = []
    if progress_callback:
        from transformers import TrainerCallback

        class UIProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    step_logs.append(logs)
                    loss = logs.get("loss", logs.get("train_loss", "?"))
                    progress_callback(
                        step_logs,
                        f"Step {state.global_step}/{state.max_steps} | Loss: {loss}"
                    )

        callbacks = [UIProgressCallback()]
    else:
        callbacks = []

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    train_result = trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    metrics["gpu"] = gpu_name
    metrics["adapter_path"] = OUTPUT_DIR
    return metrics


def main():
    print("=" * 70)
    print(f"{ACTIVE_DOMAIN.capitalize()} QLoRA Fine-Tuning")
    print("=" * 70)

    metrics = run_training()

    print(f"\nTraining complete.")
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print(f"Final loss: {metrics.get('train_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
