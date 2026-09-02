from threading import Thread
from transformers import TextIteratorStreamer, PreTrainedModel, PreTrainedTokenizerFast
from typing import Iterator, List, Dict

class Generator:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def generate_stream(self, messages: List[Dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> Iterator[str]:
        """
        Streams generated text chunks from a list of conversational messages.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for models without a chat template
            formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "
            
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p
        )
        
        # Run generation in a background thread so we can yield from the streamer in the main thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
