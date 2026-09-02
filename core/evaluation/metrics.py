import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from datasets import Dataset
import math
from tqdm import tqdm
import evaluate

def calculate_perplexity(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, eval_dataset: Dataset, batch_size: int = 4, max_length: int = 1024):
    """
    Calculates perplexity and validation loss on a canonical validation dataset.
    """
    model.eval()
    device = model.device
    
    total_loss = 0.0
    total_batches = 0
    
    # We need to process the 'messages' into actual tokens
    for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Calculating Perplexity"):
        batch_msgs = eval_dataset[i:i+batch_size]["messages"]
        
        # Format strings using tokenizer chat template
        if hasattr(tokenizer, "apply_chat_template"):
            formatted_texts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in batch_msgs]
        else:
            formatted_texts = ["\n".join([m["content"] for m in msgs]) for msgs in batch_msgs]
            
        encodings = tokenizer(
            formatted_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        # Labels are same as input_ids but we mask padding tokens with -100 so loss ignores them
        labels = encodings.input_ids.clone()
        labels[encodings.attention_mask == 0] = -100
        
        with torch.no_grad():
            outputs = model(**encodings, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1
            
    if total_batches == 0:
        return float('inf'), float('inf')
        
    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    return round(perplexity, 4), round(avg_loss, 4)

def calculate_metrics(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, eval_dataset: Dataset, batch_size: int = 4, max_length: int = 512, max_new_tokens: int = 128) -> dict:
    """
    Calculates ROUGE and BLEU metrics by generating responses and comparing with targets.
    """
    model.eval()
    device = model.device
    
    rouge = evaluate.load('rouge')
    
    all_preds = []
    all_refs = []
    
    for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Calculating Generative Metrics"):
        batch_msgs = eval_dataset[i:i+batch_size]["messages"]
        
        prompts = []
        targets = []
        
        for msgs in batch_msgs:
            # We assume the last message is from the assistant (the target)
            if msgs and msgs[-1]["role"] == "assistant":
                prompt_msgs = msgs[:-1]
                target_msg = msgs[-1]["content"]
            else:
                prompt_msgs = msgs
                target_msg = ""
                
            if hasattr(tokenizer, "apply_chat_template"):
                # apply_chat_template with add_generation_prompt=True for the prompt
                prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "\n".join([m["content"] for m in prompt_msgs]) + "\nAssistant: "
                
            prompts.append(prompt)
            targets.append(target_msg)
            
        tokenizer.padding_side = 'left' # Important for batched generation
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            
        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False, # greedy decoding for eval
            )
            
        # Decode only the generated parts
        input_len = encodings.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        all_preds.extend(preds)
        all_refs.extend(targets)
        
    rouge_results = rouge.compute(predictions=all_preds, references=all_refs)
    rougeL = round(rouge_results["rougeL"] * 100, 2)
    
    # Calculate BERTScore
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(predictions=all_preds, references=all_refs, lang="en")
    bert_f1 = sum(bert_results["f1"]) / len(bert_results["f1"]) if bert_results["f1"] else 0.0
    bert_f1 = round(bert_f1 * 100, 2)
    
    # Calculate Task-specific Accuracy / F1 (Token overlap F1)
    def compute_f1(pred, gold):
        pred_toks = set(pred.lower().split())
        gold_toks = set(gold.lower().split())
        if not pred_toks or not gold_toks:
            return int(pred_toks == gold_toks)
        common = pred_toks.intersection(gold_toks)
        if not common:
            return 0.0
        p = len(common) / len(pred_toks)
        r = len(common) / len(gold_toks)
        return 2 * (p * r) / (p + r)
        
    f1_scores = [compute_f1(p, r) for p, r in zip(all_preds, all_refs)]
    avg_f1 = round((sum(f1_scores) / len(f1_scores)) * 100, 2) if f1_scores else 0.0
    
    return {
        "ROUGE-L": rougeL,
        "BERTScore": bert_f1,
        "Task-specific Accuracy / F1": avg_f1
    }
