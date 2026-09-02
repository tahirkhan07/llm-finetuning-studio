from datasets import load_dataset as hf_load_dataset, Dataset
from pathlib import Path
from typing import Optional

def load_dataset(source: str, **kwargs) -> Dataset:
    """
    Loads a dataset from either a HuggingFace hub identifier or a local file path.
    Supports CSV, JSON, and JSONL.
    """
    if source.startswith("hf://"):
        # Remote HuggingFace dataset
        dataset_name = source[5:]
        return hf_load_dataset(dataset_name, split="train", **kwargs)
        
    path = Path(source)
    if not path.exists():
        # Maybe it's a direct HF identifier without hf://
        try:
            return hf_load_dataset(source, split="train", **kwargs)
        except Exception as e:
            raise ValueError(f"Could not load dataset from {source}. Ensure it is a valid path or HF dataset.") from e

    # Local file
    ext = path.suffix.lower()
    if ext == ".csv":
        return hf_load_dataset("csv", data_files=str(path), split="train", **kwargs)
    elif ext in [".json", ".jsonl"]:
        return hf_load_dataset("json", data_files=str(path), split="train", **kwargs)
    elif ext == ".parquet":
        return hf_load_dataset("parquet", data_files=str(path), split="train", **kwargs)
    else:
        raise ValueError(f"Unsupported local file extension: {ext}")
