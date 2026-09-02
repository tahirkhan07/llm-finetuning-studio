import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import MODEL_NAME, OUTPUT_DIR, SYSTEM_PROMPT, ACTIVE_DOMAIN


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        OUTPUT_DIR,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    return model, tokenizer


from guardrails import validate_query

def generate_response(question):
    if not validate_query(question):
        return f"\n[GUARDRAIL BLOCKED] This question appears to be outside the scope of my {ACTIVE_DOMAIN} domain expertise. Please ask a {ACTIVE_DOMAIN} related question!"

    model, tokenizer = load_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )


if __name__ == "__main__":
    question = input(f"\n{ACTIVE_DOMAIN.capitalize()} question: ").strip()

    if not question:
        raise ValueError("Please enter a question.")

    print("\n" + "=" * 70)
    print("MODEL RESPONSE")
    print("=" * 70)
    print(generate_response(question))
