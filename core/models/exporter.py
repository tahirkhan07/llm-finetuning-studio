"""
Model Exporter — Merge LoRA adapters, convert to GGUF, and push to HuggingFace Hub.

All operations are designed to be called from background threads with progress callbacks
for streaming status updates to the Gradio UI.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List

import torch


# ── Type alias for progress callbacks ────────────────────────────────────────
ProgressFn = Optional[Callable[[str], None]]

# Where we cache the llama.cpp conversion tools
_TOOLS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "llm-studio", "llama-cpp-tools")


def _log(progress_fn: ProgressFn, msg: str):
    """Send a progress message if a callback is provided."""
    if progress_fn:
        progress_fn(msg)


def _estimate_model_size_bytes(adapter_path: str) -> int:
    """
    Estimate the merged model size in bytes by reading the adapter config
    to find the base model, then checking its config for parameter count.
    Falls back to a conservative estimate if config isn't available.
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No adapter_config.json found at '{adapter_path}'. "
            f"Are you sure this is a valid PEFT adapter directory?"
        )

    with open(config_path, "r") as f:
        adapter_cfg = json.load(f)

    base_model_id = adapter_cfg.get("base_model_name_or_path", "")

    # Try to read the base model's config.json for num_parameters
    # This is a rough estimate — actual size depends on architecture
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
        # Common heuristic: hidden_size * num_layers * 12 * 2 (fp16 bytes)
        # More accurate: use the reported param count if available
        vocab_size = getattr(config, "vocab_size", 32000)
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)

        # Rough parameter count estimate
        embed_params = vocab_size * hidden_size * 2  # input + output embeddings
        attn_params = num_layers * (4 * hidden_size * hidden_size)  # Q, K, V, O
        ffn_params = num_layers * (3 * hidden_size * intermediate_size)  # gate, up, down
        total_params = embed_params + attn_params + ffn_params

        # fp16 = 2 bytes per param
        return total_params * 2
    except Exception:
        # Conservative fallback: assume 7B params in fp16 ≈ 14GB
        return 14 * 1024**3


def merge_adapter(
    adapter_path: str,
    output_dir: str,
    base_model_id: Optional[str] = None,
    progress_fn: ProgressFn = None,
) -> Dict[str, Any]:
    """
    Merge a LoRA adapter back into the base model and save as a full fp16 model.

    Args:
        adapter_path: Path to the PEFT adapter directory (must contain adapter_config.json)
        output_dir: Where to save the merged model
        base_model_id: HuggingFace model ID for the base model.
                       If None, reads from adapter_config.json.
        progress_fn: Optional callback for streaming progress messages.

    Returns:
        Dict with output_dir, size_gb, and num_shards.

    Raises:
        FileNotFoundError: If adapter_config.json is missing
        RuntimeError: If disk space is insufficient
    """
    # ── Validate adapter path ────────────────────────────────────────────
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No adapter_config.json found at '{adapter_path}'. "
            f"This doesn't appear to be a valid PEFT adapter directory."
        )

    with open(config_path, "r") as f:
        adapter_cfg = json.load(f)

    if not base_model_id:
        base_model_id = adapter_cfg.get("base_model_name_or_path", "")
        if not base_model_id:
            raise ValueError(
                "Could not determine base model ID. "
                "It's not in adapter_config.json and wasn't provided explicitly."
            )

    _log(progress_fn, f"📋 Base model: `{base_model_id}`")
    _log(progress_fn, f"📋 Adapter: `{adapter_path}`")

    # ── Pre-flight: disk space check ─────────────────────────────────────
    estimated_bytes = _estimate_model_size_bytes(adapter_path)
    estimated_gb = estimated_bytes / (1024**3)
    _log(progress_fn, f"📏 Estimated merged model size: ~{estimated_gb:.1f} GB")

    os.makedirs(output_dir, exist_ok=True)
    disk_usage = shutil.disk_usage(output_dir)
    free_gb = disk_usage.free / (1024**3)
    _log(progress_fn, f"💾 Available disk space: {free_gb:.1f} GB")

    # Need ~2x estimated size for safety (model in RAM + writing to disk)
    required_gb = estimated_gb * 2
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space. Need ~{required_gb:.1f} GB free, "
            f"but only {free_gb:.1f} GB available at '{output_dir}'. "
            f"Free up disk space or choose a different output directory."
        )

    # ── Load base model on CPU (avoids GPU OOM) ─────────────────────────
    _log(progress_fn, "⏳ Loading base model on CPU (this may take a few minutes)...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=True, token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        token=True,
    )

    _log(progress_fn, "✅ Base model loaded.")

    # ── Load and merge adapter ───────────────────────────────────────────
    _log(progress_fn, "⏳ Loading LoRA adapter and merging weights...")

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    _log(progress_fn, "✅ Adapter merged successfully.")

    # ── Save merged model ────────────────────────────────────────────────
    _log(progress_fn, f"⏳ Saving merged model to `{output_dir}`...")

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    _log(progress_fn, "✅ Merged model saved.")

    # ── Cleanup RAM ──────────────────────────────────────────────────────
    del model, base_model
    gc.collect()

    # ── Calculate actual output size ─────────────────────────────────────
    total_size = sum(
        f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file()
    )
    size_gb = total_size / (1024**3)
    num_shards = len(list(Path(output_dir).glob("model*.safetensors")))

    _log(progress_fn, f"📦 Final size: {size_gb:.2f} GB ({num_shards} shard(s))")
    _log(progress_fn, "🎉 **Merge complete!**")

    return {
        "output_dir": output_dir,
        "size_gb": round(size_gb, 2),
        "num_shards": num_shards,
        "base_model_id": base_model_id,
    }


def _ensure_gguf_converter() -> str:
    """
    Ensure the llama.cpp convert_hf_to_gguf.py script is available.
    Clones the llama.cpp repository if not cached.
    Returns the path to the script.
    """
    os.makedirs(_TOOLS_CACHE_DIR, exist_ok=True)
    repo_dir = os.path.join(_TOOLS_CACHE_DIR, "llama.cpp")
    script_path = os.path.join(repo_dir, "convert_hf_to_gguf.py")

    if os.path.exists(script_path):
        return script_path

    # Clone the repository
    try:
        subprocess.check_call(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", repo_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to clone llama.cpp repository.\n"
            f"Error: {e}\n\n"
            f"Manual fix: Clone llama.cpp manually to:\n"
            f"  {repo_dir}"
        )
        
    # Install the requirements for the conversion script
    try:
        subprocess.check_call(
            ["uv", "pip", "install", "-r", os.path.join(repo_dir, "requirements", "requirements-convert_hf_to_gguf.txt")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # Fallback to general requirements.txt just in case
        try:
            subprocess.check_call(
                ["uv", "pip", "install", "-r", os.path.join(repo_dir, "requirements.txt")],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    return script_path


SUPPORTED_GGUF_QUANT_TYPES = ["F16", "Q4_K_M", "Q5_K_M", "Q8_0"]


def convert_to_gguf(
    merged_model_dir: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    progress_fn: ProgressFn = None,
) -> Dict[str, Any]:
    """
    Convert a merged HuggingFace model to GGUF format using llama.cpp tools.

    Args:
        merged_model_dir: Path to the merged model directory (must contain config.json)
        output_path: Output .gguf file path
        quant_type: Quantization type (F16, Q4_K_M, Q5_K_M, Q8_0)
        progress_fn: Optional callback for streaming progress messages.

    Returns:
        Dict with output_path and size_gb.

    Raises:
        FileNotFoundError: If model directory is invalid
        RuntimeError: If conversion fails
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    config_file = os.path.join(merged_model_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"No config.json found at '{merged_model_dir}'. "
            f"This doesn't appear to be a valid HuggingFace model directory. "
            f"Did you run 'Merge Adapter' first?"
        )

    if quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quantization type '{quant_type}'. "
            f"Supported: {', '.join(SUPPORTED_GGUF_QUANT_TYPES)}"
        )

    # ── Pre-flight: disk space check ─────────────────────────────────────
    model_size = sum(
        f.stat().st_size for f in Path(merged_model_dir).rglob("*") if f.is_file()
    )
    # GGUF Q4 is roughly 25-30% of fp16 size, but we need space during conversion
    output_parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_parent, exist_ok=True)
    disk_usage = shutil.disk_usage(output_parent)
    free_gb = disk_usage.free / (1024**3)
    model_gb = model_size / (1024**3)

    _log(progress_fn, f"📏 Source model size: {model_gb:.2f} GB")
    _log(progress_fn, f"💾 Available disk space: {free_gb:.1f} GB")

    if free_gb < model_gb:
        raise RuntimeError(
            f"Insufficient disk space for GGUF conversion. "
            f"Need at least {model_gb:.1f} GB free, but only {free_gb:.1f} GB available."
        )

    # ── Ensure conversion script is available ────────────────────────────
    _log(progress_fn, "🔧 Checking GGUF conversion tools...")

    try:
        converter_script = _ensure_gguf_converter()
        _log(progress_fn, "✅ Conversion tools ready.")
    except RuntimeError as e:
        _log(progress_fn, f"❌ {e}")
        raise

    # ── Ensure the 'gguf' Python package is installed ────────────────────
    try:
        import gguf  # noqa: F401
    except ImportError:
        _log(progress_fn, "📦 Installing `gguf` Python package...")
        subprocess.check_call(
            ["uv", "pip", "install", "gguf"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # ── Run conversion ───────────────────────────────────────────────────
    _log(progress_fn, f"⏳ Converting to GGUF ({quant_type})... This may take several minutes.")

    cmd = [
        sys.executable,
        converter_script,
        merged_model_dir,
        "--outfile", output_path,
        "--outtype", quant_type.lower().replace("_", "-") if quant_type != "F16" else "f16",
    ]

    # For quantized types, the convert script uses f16 as base
    # Additional quantization is done via llama-quantize (separate step)
    # The convert_hf_to_gguf.py script supports: f32, f16, bf16, q8_0, auto
    # For Q4_K_M, Q5_K_M we first convert to F16, then note for user
    needs_secondary_quant = quant_type in ("Q4_K_M", "Q5_K_M")

    if needs_secondary_quant:
        # Convert to F16 first
        cmd = [
            sys.executable,
            converter_script,
            merged_model_dir,
            "--outfile", output_path,
            "--outtype", "f16",
        ]
        _log(
            progress_fn,
            f"ℹ️ Converting to F16 GGUF first. For {quant_type} quantization, "
            f"use `llama-quantize` from llama.cpp on the output file."
        )

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            if line:
                _log(progress_fn, f"  {line}")

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f"GGUF conversion failed with exit code {process.returncode}. "
                f"Check the output above for details."
            )

    except FileNotFoundError:
        raise RuntimeError(
            "Failed to run GGUF conversion script. "
            "Make sure Python is accessible and the conversion tools are properly installed."
        )

    # ── Report results ───────────────────────────────────────────────────
    if os.path.exists(output_path):
        output_size_gb = os.path.getsize(output_path) / (1024**3)
        _log(progress_fn, f"📦 Output: `{output_path}` ({output_size_gb:.2f} GB)")

        if needs_secondary_quant:
            _log(
                progress_fn,
                f"\n⚠️ **Note:** The output is in F16 format. To get {quant_type}, run:\n"
                f"```\nllama-quantize {output_path} {output_path.replace('.gguf', f'-{quant_type}.gguf')} {quant_type}\n```"
            )

        _log(progress_fn, "🎉 **GGUF conversion complete!**")

        return {
            "output_path": output_path,
            "size_gb": round(output_size_gb, 2),
            "quant_type": "F16" if needs_secondary_quant else quant_type,
        }
    else:
        raise RuntimeError(
            f"Conversion appeared to succeed but output file was not created at '{output_path}'."
        )


def _generate_model_card(
    base_model_id: str,
    adapter_path: str,
    repo_id: str,
) -> str:
    """Generate a HuggingFace model card README."""

    # Try to read LoRA config for metadata
    lora_rank = "unknown"
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            lora_rank = cfg.get("r", "unknown")
        except Exception:
            pass

    return f"""---
library_name: transformers
base_model: {base_model_id}
tags:
- qlora
- fine-tuned
- llm-fine-tuning-studio
license: mit
---

# {repo_id.split('/')[-1]}

This model is a fine-tuned version of [`{base_model_id}`](https://huggingface.co/{base_model_id})
using **QLoRA** (Quantized Low-Rank Adaptation).

## Training Details

| Parameter | Value |
|---|---|
| Base Model | `{base_model_id}` |
| Method | QLoRA |
| LoRA Rank | `{lora_rank}` |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

---

*Fine-tuned with [LLM Fine-Tuning Studio](https://github.com/llm-finetuning-studio)* 🚀
"""


def push_to_hub(
    model_dir: str,
    repo_id: str,
    private: bool = True,
    adapter_path: Optional[str] = None,
    base_model_id: Optional[str] = None,
    model_card_content: Optional[str] = None,
    progress_fn: ProgressFn = None,
) -> Dict[str, Any]:
    """
    Push a model directory (merged model or GGUF file) to HuggingFace Hub.

    Args:
        model_dir: Path to the model directory to upload
        repo_id: HuggingFace repo ID (e.g. 'username/model-name')
        private: Whether the repo should be private
        adapter_path: Original adapter path (for model card metadata)
        base_model_id: Base model ID (for model card metadata)
        model_card_content: Custom model card content. If None, auto-generated.
        progress_fn: Optional callback for streaming progress messages.

    Returns:
        Dict with repo_url and files_uploaded.

    Raises:
        RuntimeError: If not authenticated or lacking write permissions
    """
    from huggingface_hub import whoami, create_repo, upload_folder, HfApi

    # ── Auth check ───────────────────────────────────────────────────────
    _log(progress_fn, "🔑 Checking authentication...")
    try:
        user_info = whoami()
    except Exception:
        raise RuntimeError(
            "Not authenticated with HuggingFace Hub. "
            "Please go to the ⚙️ Settings tab and log in with your access token first."
        )

    username = user_info.get("name", "unknown")
    _log(progress_fn, f"✅ Authenticated as `{username}`")

    # ── Check write permissions ──────────────────────────────────────────
    auth_info = user_info.get("auth", {})
    access_token = auth_info.get("accessToken", {})
    token_role = access_token.get("role", "")

    if token_role == "read":
        raise RuntimeError(
            "Your HuggingFace token has **read-only** permissions. "
            "Push to Hub requires a token with **write** access.\n\n"
            "Create a write token at: https://huggingface.co/settings/tokens"
        )

    # ── Validate model directory ─────────────────────────────────────────
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: '{model_dir}'")

    # ── Create repo ──────────────────────────────────────────────────────
    _log(progress_fn, f"📦 Creating/verifying repo `{repo_id}` (private={private})...")

    try:
        repo_url_obj = create_repo(repo_id, exist_ok=True, private=private)
        repo_url = repo_url_obj.url if hasattr(repo_url_obj, 'url') else str(repo_url_obj)
    except Exception as e:
        raise RuntimeError(f"Failed to create repository: {e}")

    _log(progress_fn, f"✅ Repository ready: {repo_url}")

    # ── Generate model card if not provided ──────────────────────────────
    if model_card_content is None and base_model_id:
        model_card_content = _generate_model_card(
            base_model_id=base_model_id,
            adapter_path=adapter_path or model_dir,
            repo_id=repo_id,
        )

    # Write model card to the model directory
    if model_card_content:
        readme_path = os.path.join(model_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card_content)
        _log(progress_fn, "📝 Model card generated.")

    # ── Upload ───────────────────────────────────────────────────────────
    _log(progress_fn, "⏳ Uploading model files... This may take a while for large models.")

    try:
        api = HfApi()
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
        )
    except Exception as e:
        raise RuntimeError(f"Upload failed: {e}")

    # Count uploaded files
    files_uploaded = sum(1 for f in Path(model_dir).rglob("*") if f.is_file())

    _log(progress_fn, f"✅ Uploaded {files_uploaded} files.")
    _log(progress_fn, f"🎉 **Push complete!** View your model: {repo_url}")

    return {
        "repo_url": repo_url,
        "files_uploaded": files_uploaded,
    }


def list_merged_models(root_dir: str = "./outputs/merged") -> List[str]:
    """List all merged model directories."""
    root = Path(root_dir)
    if not root.exists():
        return []
    return [
        str(d) for d in root.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    ]


def list_gguf_files(root_dir: str = "./outputs") -> List[str]:
    """List all .gguf files in the outputs directory tree."""
    root = Path(root_dir)
    if not root.exists():
        return []
    return [str(f) for f in root.rglob("*.gguf")]
