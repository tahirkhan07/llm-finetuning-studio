from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from .engine import TrainingEngine
from .callbacks import ProgressCallback
import torch
import os

class SFTEngine(TrainingEngine):
    """
    Standard LoRA Engine (Full precision or mixed precision, no bitsandbytes quantization).
    """
    def prepare(self) -> None:
        # Prevent memory corruption: if the model already has adapters, unload them
        if hasattr(self.model, "unload"):
            import logging
            logging.warning("Unloading existing PEFT adapters from model.")
            self.model = self.model.unload()

        # We don't call prepare_model_for_kbit_training for standard SFT
        self.peft_config = LoraConfig(
            r=self.cfg.lora.rank,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, self.peft_config)
        
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def train(self) -> None:
        output_dir = os.path.join(self.cfg.output_dir, self.experiment_id)

        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.cfg.per_device_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            num_train_epochs=self.cfg.num_epochs,
            max_length=self.cfg.max_seq_length,
            bf16=(self.cfg.precision == "bf16"),
            fp16=(self.cfg.precision == "fp16"),
            logging_steps=10,
            report_to="none",
        )

        train_ds = self.dataset["train"] if isinstance(self.dataset, dict) and "train" in self.dataset else self.dataset
        eval_ds = self.dataset.get("validation") if isinstance(self.dataset, dict) else None

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=sft_config,
            callbacks=[ProgressCallback(self.log_path)],
        )

        try:
            self.trainer.train()
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError("CUDA Out of Memory during training! Try reducing batch size or max sequence length.") from e

    def evaluate(self) -> dict:
        if hasattr(self, "trainer") and self.trainer.eval_dataset is not None:
            return self.trainer.evaluate()
        return {}

    def save(self) -> None:
        save_path = os.path.join(self.cfg.output_dir, self.experiment_id, "adapter")
        if hasattr(self, "trainer"):
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
