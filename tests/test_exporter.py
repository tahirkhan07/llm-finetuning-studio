"""
Unit tests for core/models/exporter.py

Tests cover error paths and pre-flight checks without requiring GPU or real models.
"""

import json
import os
import shutil
import pytest
from collections import namedtuple
from unittest.mock import patch, MagicMock


# ── Test: merge_adapter validates adapter path ────────────────────────────────

def test_merge_validates_adapter_path(tmp_path):
    """merge_adapter should raise FileNotFoundError when adapter_config.json is missing."""
    from core.models.exporter import merge_adapter

    fake_adapter = str(tmp_path / "nonexistent_adapter")
    os.makedirs(fake_adapter, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        merge_adapter(
            adapter_path=fake_adapter,
            output_dir=str(tmp_path / "output"),
        )


def test_merge_validates_missing_base_model_id(tmp_path):
    """merge_adapter should raise ValueError when base model can't be determined."""
    from core.models.exporter import merge_adapter

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    # Write adapter_config.json with empty base_model_name_or_path
    config = {"base_model_name_or_path": "", "r": 16}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Could not determine base model ID"):
        merge_adapter(
            adapter_path=str(adapter_dir),
            output_dir=str(tmp_path / "output"),
        )


def test_merge_checks_disk_space(tmp_path):
    """merge_adapter should raise RuntimeError when disk space is insufficient."""
    from core.models.exporter import merge_adapter

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    config = {"base_model_name_or_path": "test/model", "r": 16}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

    output_dir = str(tmp_path / "output")

    # Create a proper namedtuple for disk_usage result
    DiskUsage = namedtuple("usage", ["total", "used", "free"])
    fake_usage = DiskUsage(total=1024**3, used=1024**3 - 1024**2, free=1024**2)

    with patch("shutil.disk_usage", return_value=fake_usage), \
         patch("core.models.exporter._estimate_model_size_bytes", return_value=10 * 1024**3):
        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            merge_adapter(
                adapter_path=str(adapter_dir),
                output_dir=output_dir,
                base_model_id="test/model",
            )


def test_merge_loads_on_cpu(tmp_path):
    """merge_adapter should load the base model on CPU to avoid GPU OOM."""
    from core.models.exporter import merge_adapter

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    config = {"base_model_name_or_path": "test/model", "r": 16}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

    output_dir = str(tmp_path / "output")

    # Mock everything to inspect the device_map argument
    mock_model = MagicMock()
    mock_model.merge_and_unload.return_value = mock_model
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "test"

    mock_peft_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_model

    # Patch at the source locations (where they're imported from)
    with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_auto, \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("peft.PeftModel.from_pretrained", return_value=mock_peft_model), \
         patch("core.models.exporter._estimate_model_size_bytes", return_value=1024):

        merge_adapter(
            adapter_path=str(adapter_dir),
            output_dir=output_dir,
            base_model_id="test/model",
        )

        # Verify device_map="cpu" was passed
        call_kwargs = mock_auto.call_args
        assert call_kwargs.kwargs.get("device_map") == "cpu", \
            "Base model must be loaded with device_map='cpu' to avoid GPU OOM during merge"


# ── Test: push_to_hub requires authentication ────────────────────────────────

def test_push_requires_auth(tmp_path):
    """push_to_hub should raise RuntimeError when not authenticated."""
    from core.models.exporter import push_to_hub

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with patch("huggingface_hub.whoami", side_effect=Exception("Not logged in")):
        with pytest.raises(RuntimeError, match="Not authenticated"):
            push_to_hub(
                model_dir=str(model_dir),
                repo_id="test-user/test-model",
            )


def test_push_requires_write_token(tmp_path):
    """push_to_hub should raise RuntimeError when token is read-only."""
    from core.models.exporter import push_to_hub

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    fake_user_info = {
        "name": "test-user",
        "auth": {
            "accessToken": {
                "role": "read",
            }
        }
    }

    with patch("huggingface_hub.whoami", return_value=fake_user_info):
        with pytest.raises(RuntimeError, match="read-only"):
            push_to_hub(
                model_dir=str(model_dir),
                repo_id="test-user/test-model",
            )


# ── Test: convert_to_gguf validates model directory ──────────────────────────

def test_gguf_validates_model_dir(tmp_path):
    """convert_to_gguf should raise FileNotFoundError when config.json is missing."""
    from core.models.exporter import convert_to_gguf

    empty_dir = str(tmp_path / "empty_model")
    os.makedirs(empty_dir, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="config.json"):
        convert_to_gguf(
            merged_model_dir=empty_dir,
            output_path=str(tmp_path / "output.gguf"),
        )


def test_gguf_rejects_invalid_quant_type(tmp_path):
    """convert_to_gguf should raise ValueError for unsupported quant types."""
    from core.models.exporter import convert_to_gguf

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="Unsupported quantization type"):
        convert_to_gguf(
            merged_model_dir=str(model_dir),
            output_path=str(tmp_path / "output.gguf"),
            quant_type="INVALID_TYPE",
        )


# ── Test: list_merged_models discovery ────────────────────────────────────────

def test_list_merged_models(tmp_path):
    """list_merged_models should find directories containing config.json."""
    from core.models.exporter import list_merged_models

    # Create a valid merged model directory
    valid = tmp_path / "model_a"
    valid.mkdir()
    (valid / "config.json").write_text("{}")

    # Create an invalid directory (no config.json)
    invalid = tmp_path / "model_b"
    invalid.mkdir()
    (invalid / "random.txt").write_text("not a model")

    results = list_merged_models(str(tmp_path))
    assert len(results) == 1
    assert "model_a" in results[0]


def test_list_merged_models_empty(tmp_path):
    """list_merged_models should return empty list for nonexistent directory."""
    from core.models.exporter import list_merged_models

    results = list_merged_models(str(tmp_path / "nonexistent"))
    assert results == []
