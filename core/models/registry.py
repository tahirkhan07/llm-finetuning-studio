# A registry of recommended models supported by the backend.
# Users can always paste ANY HuggingFace model ID into the dropdown — this list
# is only for convenience.

POPULAR_MODELS = [
    # --- Qwen (Open, no auth required) ---
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    # --- Microsoft Phi (Open, no auth required) ---
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    # --- TinyLlama (Open, great for testing) ---
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # --- Mistral (Open, no auth required) ---
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # --- Google Gemma (Gated — requires HF login) ---
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-3-4b-it",
    # --- Meta Llama (Gated — requires HF login) ---
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]
