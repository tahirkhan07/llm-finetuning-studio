from pydantic import BaseModel
from typing import Dict, Any, List
from datasets import Dataset

class ValidationReport(BaseModel):
    total_samples: int
    valid_samples: int
    invalid_samples: int
    duplicate_samples: int
    avg_tokens: float = 0.0
    max_tokens: int = 0
    errors: List[str] = []

def validate_dataset(dataset: Dataset, tokenizer: Any = None) -> ValidationReport:
    """
    Validates a canonical dataset (with 'messages' column).
    """
    report = ValidationReport(
        total_samples=len(dataset),
        valid_samples=0,
        invalid_samples=0,
        duplicate_samples=0
    )
    
    seen_hashes = set()
    errors = []
    valid_count = 0
    
    if "messages" not in dataset.column_names:
        report.errors.append("Dataset missing required 'messages' column. Did you run the mapper?")
        report.invalid_samples = len(dataset)
        return report

    # Optional token stats
    token_lengths = []
    
    for i, row in enumerate(dataset):
        messages = row.get("messages", [])
        
        # Check structure
        if not isinstance(messages, list) or len(messages) == 0:
            errors.append(f"Row {i} has empty or invalid messages.")
            continue
            
        # Check empty strings
        is_valid = True
        for msg in messages:
            if not msg.get("content") or not isinstance(msg["content"], str) or msg["content"].strip() == "":
                errors.append(f"Row {i} has an empty content string.")
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Check duplicates
        # hash based on user + assistant content
        content_str = "".join([m["content"] for m in messages])
        row_hash = hash(content_str)
        if row_hash in seen_hashes:
            report.duplicate_samples += 1
            # We don't necessarily count duplicates as invalid, but let's track them
        seen_hashes.add(row_hash)
        
        valid_count += 1
        
        # Optional: compute token lengths if tokenizer provided
        if tokenizer:
            try:
                # Approximate token length (applying chat template if it exists)
                if hasattr(tokenizer, "apply_chat_template"):
                    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                    token_lengths.append(len(tokens))
                else:
                    tokens = tokenizer.encode(content_str)
                    token_lengths.append(len(tokens))
            except Exception:
                pass

    report.valid_samples = valid_count
    report.invalid_samples = report.total_samples - valid_count
    
    # Take first 100 errors to prevent overwhelming
    report.errors = errors[:100]
    
    if token_lengths:
        report.avg_tokens = sum(token_lengths) / len(token_lengths)
        report.max_tokens = max(token_lengths)

    return report
