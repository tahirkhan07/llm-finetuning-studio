from datasets import load_dataset

from config import (
    DATASET_NAME,
    MAX_TRAIN_SAMPLES,
    SEED,
    VALIDATION_SIZE,
    SYSTEM_PROMPT,
    DATA_FILES,
    DATASET_COLUMNS,
)


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def create_user_message(instruction, input_text):
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)

    if not input_text or input_text.lower() in {
        "<noinput>",
        "no input",
        "none",
        "nan",
    }:
        return instruction

    return f"{instruction}\n\nAdditional context:\n{input_text}"


def convert_to_messages(example):
    inst_col = DATASET_COLUMNS["instruction"]
    in_col = DATASET_COLUMNS["input"]
    out_col = DATASET_COLUMNS["output"]

    instruction = example.get(inst_col, "") if inst_col else ""
    input_text = example.get(in_col, "") if in_col else ""
    output_text = example.get(out_col, "") if out_col else ""

    user_text = create_user_message(instruction, input_text)
    assistant_text = clean_text(output_text)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def load_domain_dataset(max_samples=None):
    """
    Load and preprocess the dataset for the active domain.
    max_samples: if set, overrides the config MAX_TRAIN_SAMPLES.
    """
    print(f"Loading dataset: {DATASET_NAME}")

    load_kwargs = {"path": DATASET_NAME, "split": "train"}
    if DATA_FILES:
        load_kwargs["data_files"] = DATA_FILES

    dataset = load_dataset(**load_kwargs)

    print(f"Original examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    inst_col = DATASET_COLUMNS["instruction"]
    out_col = DATASET_COLUMNS["output"]

    required = set()
    if inst_col:
        required.add(inst_col)
    if out_col:
        required.add(out_col)

    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    dataset = dataset.filter(
        lambda x: (
            x.get(inst_col) is not None
            and x.get(out_col) is not None
            and len(clean_text(x.get(inst_col))) > 0
            and len(clean_text(x.get(out_col))) > 0
        )
    )

    dataset = dataset.shuffle(seed=SEED)

    # max_samples arg from UI overrides config setting
    limit = max_samples if max_samples is not None else MAX_TRAIN_SAMPLES
    if limit is not None:
        dataset = dataset.select(
            range(min(limit, len(dataset)))
        )

    dataset = dataset.map(
        convert_to_messages,
        remove_columns=dataset.column_names,
    )

    split = dataset.train_test_split(
        test_size=VALIDATION_SIZE,
        seed=SEED,
    )

    print(f"Train examples: {len(split['train'])}")
    print(f"Validation examples: {len(split['test'])}")

    return split["train"], split["test"]
