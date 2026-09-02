MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ACTIVE_DOMAIN = "coding"  # Change this to switch domains ("medical", "coding", "law")

DOMAINS = {
    "medical": {
        "dataset": "casey-martin/MedInstruct",
        "data_files": "data/MedInstruct-52k.json",
        "system_prompt": (
            "You are a medical information assistant. "
            "Provide clear, educational, evidence-aware medical information. "
            "Do not present yourself as a doctor and do not replace professional "
            "medical advice. When appropriate, recommend consulting a qualified "
            "healthcare professional. "
            "IMPORTANT GUARDRAIL: You must ONLY answer questions related to medicine and healthcare. "
            "If a question is outside this domain, politely refuse to answer."
        ),
        "output_dir": "./outputs/medical-qwen-lora",
        "dataset_columns": {"instruction": "instruction", "input": "input", "output": "output"}
    },
    "coding": {
        "dataset": "m-a-p/CodeFeedback-Filtered-Instruction",
        "data_files": None,
        "system_prompt": (
            "You are an expert software engineer and helpful programming assistant. "
            "Provide clean, efficient, and well-documented code. Explain your logic clearly. "
            "IMPORTANT GUARDRAIL: You must ONLY answer questions related to programming, software development, and technology. "
            "If a question is outside this domain, politely refuse to answer."
        ),
        "output_dir": "./outputs/coding-qwen-lora",
        "dataset_columns": {"instruction": "query", "input": None, "output": "answer"}
    },
    "law": {
        # Equall/Saul-Instruct-v1 is the official dataset behind SaulLM, a legal domain LLM.
        # It uses instruction/input/output schema and is publicly accessible.
        "dataset": "Equall/Saul-Instruct-v1",
        "data_files": None,
        "system_prompt": (
            "You are a knowledgeable legal assistant. Provide accurate, educational legal information. "
            "Do not provide official legal advice or replace a licensed attorney. "
            "Always recommend consulting a lawyer for formal legal counsel. "
            "IMPORTANT GUARDRAIL: You must ONLY answer questions related to law and legal matters. "
            "If a question is outside this domain, politely refuse to answer."
        ),
        "output_dir": "./outputs/law-qwen-lora",
        "dataset_columns": {"instruction": "instruction", "input": "input", "output": "output"}
    },
    "finance": {
        # gbharti/finance-alpaca is a well-known freely accessible finance instruction dataset.
        "dataset": "gbharti/finance-alpaca",
        "data_files": None,
        "system_prompt": (
            "You are an expert financial assistant. Provide clear, educational financial information "
            "covering topics like investments, markets, economics, and personal finance. "
            "Do not provide personalized financial advice or replace a certified financial advisor. "
            "IMPORTANT GUARDRAIL: You must ONLY answer questions related to finance and economics. "
            "If a question is outside this domain, politely refuse to answer."
        ),
        "output_dir": "./outputs/finance-qwen-lora",
        "dataset_columns": {"instruction": "instruction", "input": "input", "output": "output"}
    }
}

DATASET_NAME = DOMAINS[ACTIVE_DOMAIN]["dataset"]
DATA_FILES = DOMAINS[ACTIVE_DOMAIN]["data_files"]
SYSTEM_PROMPT = DOMAINS[ACTIVE_DOMAIN]["system_prompt"]
OUTPUT_DIR = DOMAINS[ACTIVE_DOMAIN]["output_dir"]
DATASET_COLUMNS = DOMAINS[ACTIVE_DOMAIN]["dataset_columns"]
SEED = 42

# Start small to verify the complete pipeline.
MAX_TRAIN_SAMPLES = 500

# Keep a small validation set.
VALIDATION_SIZE = 0.05

MAX_SEQ_LENGTH = 1024

# QLoRA / LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
