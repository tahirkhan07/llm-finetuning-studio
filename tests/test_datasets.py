import pytest
from datasets import Dataset
from core.datasets.mapper import map_to_canonical
from core.datasets.validator import validate_dataset
from core.datasets.splitter import split_dataset

@pytest.fixture
def sample_dataset():
    data = {
        "instruction": ["Translate to French.", "What is 2+2?", "Write a poem."],
        "input": ["Hello", "", "About nature"],
        "output": ["Bonjour", "4", "Trees are green..."]
    }
    return Dataset.from_dict(data)

def test_map_to_canonical(sample_dataset):
    mapped = map_to_canonical(sample_dataset, "instruction", "input", "output", "System prompt")
    
    assert "messages" in mapped.column_names
    assert len(mapped) == 3
    
    msgs = mapped[0]["messages"]
    assert len(msgs) == 3 # System, user, assistant
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "System prompt"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Translate to French.\n\nHello"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "Bonjour"
    
    # Test row with empty input
    msgs_no_input = mapped[1]["messages"]
    assert msgs_no_input[1]["role"] == "user"
    assert msgs_no_input[1]["content"] == "What is 2+2?"

def test_validate_dataset():
    # Valid data
    valid_data = Dataset.from_dict({
        "messages": [
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]
        ]
    })
    
    report = validate_dataset(valid_data)
    assert report.total_samples == 2
    assert report.valid_samples == 2
    assert report.invalid_samples == 0
    
    # Invalid data (empty string)
    invalid_data = Dataset.from_dict({
        "messages": [
            [{"role": "user", "content": ""}, {"role": "assistant", "content": "hello"}]
        ]
    })
    
    report2 = validate_dataset(invalid_data)
    assert report2.valid_samples == 0
    assert report2.invalid_samples == 1

def test_split_dataset(sample_dataset):
    # Need a slightly larger dataset for splits to not be empty, so we replicate it
    large_data = {
        "text": ["dummy"] * 100
    }
    ds = Dataset.from_dict(large_data)
    
    splits = split_dataset(ds, train_ratio=0.8, val_ratio=0.1, seed=42)
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    
    # Total is 100
    assert len(splits["train"]) == 80
    assert len(splits["validation"]) == 10
    assert len(splits["test"]) == 10


def test_map_to_canonical_list_columns():
    """Columns with list values (e.g. tags) should be coerced to strings."""
    data = {
        "quote": ["Be the change.", "Stay hungry."],
        "tags": [["inspirational", "life"], ["motivation", "tech"]],
    }
    ds = Dataset.from_dict(data)
    mapped = map_to_canonical(ds, "quote", None, "tags", "You are helpful.")
    
    assert "messages" in mapped.column_names
    msgs = mapped[0]["messages"]
    # Assistant content should be a string, not a list
    assert msgs[2]["role"] == "assistant"
    assert isinstance(msgs[2]["content"], str)
    assert "inspirational" in msgs[2]["content"]
    assert "life" in msgs[2]["content"]


def test_map_to_canonical_preformatted():
    """Datasets with an existing 'messages' column should be auto-detected and passed through."""
    data = {
        "messages": [
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "cya"}],
        ],
        "extra_col": ["a", "b"],
    }
    ds = Dataset.from_dict(data)
    mapped = map_to_canonical(ds, "extra_col", None, "extra_col")
    
    # Should keep the original messages and drop extra columns
    assert "messages" in mapped.column_names
    assert "extra_col" not in mapped.column_names
    assert mapped[0]["messages"][0]["content"] == "hi"

def test_map_to_canonical_sharegpt():
    """Datasets with ShareGPT 'from'/'value' format should be auto-converted to 'role'/'content'."""
    data = {
        "conversations": [
            [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}],
            [{"from": "system", "value": "sys"}, {"from": "human", "value": "bye"}, {"from": "gpt", "value": "cya"}],
        ],
    }
    ds = Dataset.from_dict(data)
    mapped = map_to_canonical(ds, "", None, "")
    
    assert "messages" in mapped.column_names
    assert "conversations" not in mapped.column_names
    
    msgs = mapped[0]["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["role"] == "assistant"
    
    msgs2 = mapped[1]["messages"]
    assert msgs2[0]["role"] == "system"
    assert msgs2[0]["content"] == "sys"
