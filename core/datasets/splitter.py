import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Tuple

def deduplicate_dataset(dataset: Dataset) -> Dataset:
    """
    Removes exact duplicate rows from the dataset to prevent data leakage 
    between train, validation, and test splits.
    """
    df = dataset.to_pandas()
    # Deduplicate based on all columns (usually instruction, input, output, system)
    # We only care about string columns or hashable columns
    original_len = len(df)
    df = df.drop_duplicates()
    new_len = len(df)
    
    if new_len < original_len:
        print(f"Deduplicated {original_len - new_len} samples.")
        
    return Dataset.from_pandas(df)

def split_dataset(dataset: Dataset, train_ratio: float = 0.9, val_ratio: float = 0.05, seed: int = 42) -> DatasetDict:
    """
    Splits a single dataset into train, validation, and (optionally) test splits.
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio cannot exceed 1.0")
        
    test_ratio = 1.0 - (train_ratio + val_ratio)
    
    # If the train ratio is 1.0, return just the train split
    if train_ratio >= 0.999:
        return DatasetDict({"train": dataset})
    
    # First split into train and temp (val + test)
    split1 = dataset.train_test_split(train_size=train_ratio, seed=seed)
    train_ds = split1["train"]
    temp_ds = split1["test"]
    
    if test_ratio <= 0.001 or len(temp_ds) < 2:
        # No test split or temp is too small to split further
        return DatasetDict({
            "train": train_ds,
            "validation": temp_ds
        })
        
    # Split temp into val and test
    # Adjusted ratio for the remaining portion
    adjusted_val_ratio = val_ratio / (val_ratio + test_ratio)
    
    # Protect against tiny splits crashing train_test_split
    val_size = int(len(temp_ds) * adjusted_val_ratio)
    if val_size < 1:
        val_size = 1
    elif val_size >= len(temp_ds):
        val_size = len(temp_ds) - 1
        
    split2 = temp_ds.train_test_split(train_size=val_size, seed=seed)
    
    return DatasetDict({
        "train": train_ds,
        "validation": split2["train"],
        "test": split2["test"]
    })
