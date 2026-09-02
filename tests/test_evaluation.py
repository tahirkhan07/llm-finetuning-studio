import pytest
from core.evaluation.metrics import calculate_perplexity
import torch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        
    def forward(self, input_ids, labels=None, **kwargs):
        class Output:
            loss = torch.tensor(1.0)
        return Output()
        
    def eval(self):
        pass

class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        class Encodings(dict):
            def __init__(self):
                super().__init__()
                self.input_ids = torch.tensor([[1, 2, 3]])
                self.attention_mask = torch.tensor([[1, 1, 1]])
                self["input_ids"] = self.input_ids
                self["attention_mask"] = self.attention_mask
            def to(self, device):
                return self
        return Encodings()

def test_calculate_perplexity():
    # Simple mock test
    model = DummyModel()
    tokenizer = DummyTokenizer()
    from datasets import Dataset
    dataset = Dataset.from_dict({"messages": [[{"role": "user", "content": "hi"}]]})
    
    ppl, val_loss = calculate_perplexity(model, tokenizer, dataset, batch_size=1)
    
    # exp(1.0) = 2.718
    assert round(ppl, 3) == 2.718
