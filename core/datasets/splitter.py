from datasets import Dataset, DatasetDict
from typing import Tuple

def split_dataset(dataset: Dataset, train_ratio: float = 0.9, val_ratio: float = 0.05, seed: int = 42) -> DatasetDict:
    """
    Splits a single dataset into train, validation, and (optionally) test splits.
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio cannot exceed 1.0")
        
    test_ratio = 1.0 - (train_ratio + val_ratio)
    
    # First split into train and temp (val + test)
    split1 = dataset.train_test_split(train_size=train_ratio, seed=seed)
    train_ds = split1["train"]
    temp_ds = split1["test"]
    
    if test_ratio <= 0.001:
        # No test split
        return DatasetDict({
            "train": train_ds,
            "validation": temp_ds
        })
        
    # Split temp into val and test
    # Adjusted ratio for the remaining portion
    adjusted_val_ratio = val_ratio / (val_ratio + test_ratio)
    split2 = temp_ds.train_test_split(train_size=adjusted_val_ratio, seed=seed)
    
    return DatasetDict({
        "train": train_ds,
        "validation": split2["train"],
        "test": split2["test"]
    })
