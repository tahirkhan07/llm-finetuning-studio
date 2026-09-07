import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from datasets import Dataset
import math
from tqdm import tqdm
import evaluate
import time
import os
from google import genai
from pydantic import BaseModel, Field


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    eval_dataset: Dataset,
    batch_size: int = 4,
    max_length: int = 1024
):
    """
    Calculates token-weighted perplexity and validation loss
    on the evaluation dataset.
    """

    model.eval()
    device = model.device

    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(
        range(0, len(eval_dataset), batch_size),
        desc="Calculating Perplexity"
    ):

        batch_msgs = eval_dataset[i:i + batch_size]["messages"]

        # Format messages using the model's chat template
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_texts = [
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False
                )
                for msgs in batch_msgs
            ]
        else:
            formatted_texts = [
                "\n".join(
                    [m["content"] for m in msgs]
                )
                for msgs in batch_msgs
            ]

        encodings = tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        # Labels are the same as input_ids
        labels = encodings.input_ids.clone()

        # Ignore padding tokens
        labels[encodings.attention_mask == 0] = -100

        with torch.no_grad():

            outputs = model(
                **encodings,
                labels=labels
            )

        # Hugging Face loss is averaged over valid tokens.
        # Convert it back to total loss so that every token
        # receives equal weight across the entire dataset.
        batch_loss = outputs.loss.item()

        valid_tokens = (
            labels != -100
        ).sum().item()

        if valid_tokens > 0:

            total_loss += (
                batch_loss * valid_tokens
            )

            total_tokens += valid_tokens

    if total_tokens == 0:
        return float("inf"), float("inf")

    # Correct token-weighted average loss
    avg_loss = total_loss / total_tokens

    # Perplexity = e^(average loss)
    perplexity = math.exp(avg_loss)

    return (
        round(perplexity, 4),
        round(avg_loss, 4)
    )


def calculate_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    eval_dataset: Dataset,
    batch_size: int = 4,
    max_length: int = 512,
    max_new_tokens: int = 128
) -> dict:
    """
    Calculates ROUGE, BLEU, BERTScore, token-overlap F1,
    response latency, tokens/sec, and LLM-as-a-judge scores.
    """

    model.eval()
    device = model.device

    rouge = evaluate.load("rouge")

    try:
        bleu = evaluate.load("bleu")
    except Exception:
        bleu = None

    # Load BERTScore
    bertscore = evaluate.load("bertscore")

    all_preds = []
    all_refs = []

    total_time = 0.0
    total_new_tokens = 0

    # ---------------------------------------------------------
    # Save original padding side
    # ---------------------------------------------------------

    original_padding_side = tokenizer.padding_side

    # Left padding is required for batched generation
    tokenizer.padding_side = "left"

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # Generation
    # ---------------------------------------------------------

    for i in tqdm(
        range(0, len(eval_dataset), batch_size),
        desc="Calculating Generative Metrics"
    ):

        batch_msgs = eval_dataset[i:i + batch_size]["messages"]

        prompts = []
        targets = []

        for msgs in batch_msgs:

            # Last assistant message is the target
            if (
                msgs
                and msgs[-1]["role"] == "assistant"
            ):

                prompt_msgs = msgs[:-1]
                target_msg = msgs[-1]["content"]

            else:

                prompt_msgs = msgs
                target_msg = ""

            # Create generation prompt
            if hasattr(
                tokenizer,
                "apply_chat_template"
            ):

                prompt = tokenizer.apply_chat_template(
                    prompt_msgs,
                    tokenize=False,
                    add_generation_prompt=True
                )

            else:

                prompt = (
                    "\n".join(
                        [
                            m["content"]
                            for m in prompt_msgs
                        ]
                    )
                    + "\nAssistant: "
                )

            prompts.append(prompt)
            targets.append(target_msg)

        # -----------------------------------------------------
        # Tokenization
        # -----------------------------------------------------

        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        # -----------------------------------------------------
        # Accurate GPU timing
        # -----------------------------------------------------

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        generation_time = (
            end_time - start_time
        )

        total_time += generation_time

        # -----------------------------------------------------
        # Get generated tokens
        # -----------------------------------------------------

        input_len = encodings.input_ids.shape[1]

        generated_tokens = outputs[:, input_len:]

        # -----------------------------------------------------
        # Count only actual generated tokens
        # -----------------------------------------------------

        if tokenizer.pad_token_id is not None:

            actual_generated_tokens = (
                generated_tokens
                != tokenizer.pad_token_id
            ).sum().item()

        else:

            actual_generated_tokens = (
                generated_tokens.numel()
            )

        total_new_tokens += (
            actual_generated_tokens
        )

        # -----------------------------------------------------
        # Decode predictions
        # -----------------------------------------------------

        preds = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        all_preds.extend(preds)
        all_refs.extend(targets)

    # Restore original padding configuration
    tokenizer.padding_side = original_padding_side

    # =========================================================
    # ROUGE-L
    # =========================================================

    rouge_results = rouge.compute(
        predictions=all_preds,
        references=all_refs
    )

    rougeL = round(
        rouge_results["rougeL"] * 100,
        2
    )

    # =========================================================
    # BERTScore
    # =========================================================

    bert_results = bertscore.compute(
        predictions=all_preds,
        references=all_refs,
        lang="en"
    )

    bert_f1 = (
        sum(bert_results["f1"])
        / len(bert_results["f1"])
        if bert_results["f1"]
        else 0.0
    )

    bert_f1 = round(
        bert_f1 * 100,
        2
    )

    # =========================================================
    # Task-specific Accuracy / F1
    # =========================================================

    def compute_f1(pred, gold):

        # Use tokens instead of sets so repeated words
        # are handled correctly.
        pred_toks = pred.lower().split()
        gold_toks = gold.lower().split()

        if not pred_toks or not gold_toks:

            return int(
                pred_toks == gold_toks
            )

        from collections import Counter

        pred_counter = Counter(pred_toks)
        gold_counter = Counter(gold_toks)

        common = sum(
            (pred_counter & gold_counter).values()
        )

        if common == 0:
            return 0.0

        precision = (
            common / len(pred_toks)
        )

        recall = (
            common / len(gold_toks)
        )

        if precision + recall == 0:
            return 0.0

        return (
            2 * precision * recall
        ) / (
            precision + recall
        )

    f1_scores = [
        compute_f1(p, r)
        for p, r in zip(
            all_preds,
            all_refs
        )
    ]

    avg_f1 = (
        round(
            (
                sum(f1_scores)
                / len(f1_scores)
            ) * 100,
            2
        )
        if f1_scores
        else 0.0
    )

    # =========================================================
    # BLEU
    # =========================================================

    bleu_score = 0.0

    if bleu:

        try:

            bleu_refs = [
                [ref]
                for ref in all_refs
            ]

            bleu_results = bleu.compute(
                predictions=all_preds,
                references=bleu_refs,
                smooth=True
            )

            bleu_score = round(
                bleu_results["bleu"] * 100,
                2
            )

        except Exception:

            bleu_score = 0.0

    # =========================================================
    # Response Latency
    # =========================================================

    avg_latency = (
        round(
            total_time
            / len(eval_dataset),
            4
        )
        if len(eval_dataset) > 0
        else 0.0
    )

    # =========================================================
    # Tokens Per Second
    # =========================================================

    tokens_per_sec = (
        round(
            total_new_tokens
            / total_time,
            2
        )
        if total_time > 0
        else 0.0
    )

    # =========================================================
    # LLM-as-a-Judge (Gemini)
    # =========================================================
    
    avg_correctness = 0.0
    avg_relevance = 0.0
    avg_completeness = 0.0

    if os.environ.get("GEMINI_API_KEY"):
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        print(f"LLM-as-a-Judge: Using Gemini API for semantic evaluation...")
        
        class EvaluationScore(BaseModel):
            correctness: float = Field(description="Score between 1.0 and 5.0 for factual accuracy compared to the reference.")
            relevance: float = Field(description="Score between 1.0 and 5.0 for how well the answer addresses the specific prompt.")
            completeness: float = Field(description="Score between 1.0 and 5.0 for whether the answer provides all necessary detail without hallucinating.")
        
        prompt_template = """
        You are an expert evaluator. Evaluate the following generated response against the reference response.
        Score it out of 5 for Correctness, Relevance, and Completeness.
        
        Reference Response:
        {reference}
        
        Generated Response:
        {generated}
        """
        
        total_correct = 0.0
        total_rel = 0.0
        total_comp = 0.0
        evaluated = 0
        
        # We only evaluate a subset if the dataset is large to save API costs
        sample_size = min(len(all_preds), 20)
        
        for i in range(sample_size):
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt_template.format(
                        reference=all_refs[i],
                        generated=all_preds[i]
                    ),
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=EvaluationScore,
                    ),
                )
                
                score = response.parsed
                if score:
                    total_correct += score.correctness
                    total_rel += score.relevance
                    total_comp += score.completeness
                    evaluated += 1
            except Exception as e:
                print(f"Gemini Eval Error: {e}")
                
        if evaluated > 0:
            avg_correctness = round(total_correct / evaluated, 2)
            avg_relevance = round(total_rel / evaluated, 2)
            avg_completeness = round(total_comp / evaluated, 2)

    # =========================================================
    # SAME RETURN FORMAT AS YOUR ORIGINAL CODE
    # =========================================================

    return {
        "ROUGE-L": rougeL,
        "BLEU": bleu_score,
        "BERTScore": bert_f1,
        "Task-specific Accuracy / F1": avg_f1,
        "Response latency (s)": avg_latency,
        "Tokens/sec": tokens_per_sec,
        "LLM Correctness (out of 5)": avg_correctness,
        "LLM Relevance (out of 5)": avg_relevance,
        "LLM Completeness (out of 5)": avg_completeness
    }