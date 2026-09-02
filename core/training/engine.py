from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .config import TrainingConfig
import uuid
import os

def generate_experiment_id() -> str:
    return f"EXP-{uuid.uuid4().hex[:8].upper()}"

class TrainingEngine(ABC):
    def __init__(self, cfg: TrainingConfig, model: Any, tokenizer: Any, dataset: Any):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.experiment_id = generate_experiment_id()
        self.log_dir = os.path.join("experiments", self.experiment_id)
        self.log_path = os.path.join(self.log_dir, "training_log.json")
        
        os.makedirs(self.log_dir, exist_ok=True)

    @abstractmethod
    def prepare(self) -> None:
        """Prepares the model and dataset for training."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Executes the training loop."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Runs evaluation if a validation split exists."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Saves the fine-tuned adapter/model to the output directory."""
        pass
