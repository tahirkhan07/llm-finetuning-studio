from core.hardware.detector import HardwareInfo
from core.training.config import TrainingConfig

class AppState:
    """
    Singleton object to hold the current session's state across Gradio tabs.
    Since Gradio runs a single python process in our MVP, this is sufficient.
    """
    def __init__(self):
        # Hardware
        self.hardware: HardwareInfo = None
        
        # Models
        self.model_id: str = None
        self.model = None
        self.tokenizer = None
        self.is_4bit: bool = True
        
        # Datasets
        self.raw_dataset = None
        self.canonical_dataset = None
        
        # Training
        self.training_cfg: TrainingConfig = None
        
        # Inference
        self.inference_adapter_path: str = None
        self.inference_model = None
        self.inference_tokenizer = None
        
    def clear_vram(self):
        """Clears models from VRAM to prevent OOM across tabs."""
        import torch
        import gc
        self.model = None
        self.inference_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Create the global singleton
state = AppState()
