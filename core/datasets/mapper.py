from datasets import Dataset
from typing import Dict, Optional, Any


def _to_str(value: Any) -> str:
    """
    Safely converts any column value to a string.
    Handles lists, dicts, numbers, None, booleans — anything a dataset column might contain.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # e.g. tags = ["inspirational", "love"] → "inspirational, love"
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        # Rare, but possible. Flatten to a readable string.
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    # int, float, bool, etc.
    return str(value)


def _is_preformatted(dataset: Dataset) -> bool:
    """
    Checks if the dataset already has a 'messages' or 'conversations' column
    in the standard chat format (role/content) or ShareGPT format (from/value).
    """
    for col_name in ("messages", "conversations"):
        if col_name in dataset.column_names:
            # Peek at the first row to confirm structure
            sample = dataset[0][col_name]
            if isinstance(sample, list) and len(sample) > 0:
                first_msg = sample[0]
                if isinstance(first_msg, dict):
                    if "role" in first_msg and "content" in first_msg:
                        return True
                    if "from" in first_msg and "value" in first_msg:
                        return True
    return False


def map_preformatted(dataset: Dataset) -> Dataset:
    """
    Handles datasets that already have a 'messages' or 'conversations' column.
    Renames 'conversations' → 'messages' if needed and drops other columns.
    Converts ShareGPT format (from/value) to standard format (role/content) if detected.
    """
    col_to_use = "messages" if "messages" in dataset.column_names else "conversations"
    
    # Check if it needs ShareGPT conversion
    sample_msg = dataset[0][col_to_use][0]
    needs_sharegpt_conversion = "from" in sample_msg and "value" in sample_msg

    def convert_sharegpt_row(row):
        new_msgs = []
        for msg in row[col_to_use]:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            old_role = msg.get("from", "user")
            new_msgs.append({
                "role": role_map.get(old_role, old_role),
                "content": msg.get("value", "")
            })
        return {"messages": new_msgs}

    if needs_sharegpt_conversion:
        dataset = dataset.map(convert_sharegpt_row)
    elif col_to_use != "messages":
        dataset = dataset.rename_column(col_to_use, "messages")

    # Keep only the messages column
    cols_to_remove = [c for c in dataset.column_names if c != "messages"]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    return dataset


def map_to_canonical(
    dataset: Dataset,
    instruction_col: str,
    input_col: Optional[str],
    output_col: str,
    system_prompt: Optional[str] = None,
) -> Dataset:
    """
    Maps varying dataset columns to a canonical list of messages format:
    { "messages": [{"role": "system", "content": "..."}, {"role": "user", ...}, {"role": "assistant", ...}] }

    Handles any column type (strings, lists, numbers, dicts) by coercing to string.
    Also auto-detects datasets that are already in chat format.
    """

    # Auto-detect pre-formatted datasets
    if _is_preformatted(dataset):
        return map_preformatted(dataset)

    def map_row(row: Dict[str, Any]) -> Dict[str, Any]:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Construct the user message — coerce to string to handle lists, numbers, etc.
        instruction = _to_str(row.get(instruction_col, ""))
        if input_col and row.get(input_col):
            input_text = _to_str(row[input_col])
            user_content = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction
        else:
            user_content = instruction

        messages.append({"role": "user", "content": user_content})

        # Construct the assistant message — coerce to string
        assistant_content = _to_str(row.get(output_col, ""))
        messages.append({"role": "assistant", "content": assistant_content})

        return {"messages": messages}

    # Apply the mapping and remove original columns
    mapped_dataset = dataset.map(map_row, remove_columns=dataset.column_names)
    return mapped_dataset
