from transformers import pipeline

class Guardrails:
    def __init__(self, model_id: str = "facebook/bart-large-mnli", threshold: float = 0.8):
        self.threshold = threshold
        # Initialize zero-shot classification pipeline (lazy load recommended for real apps, but we initialize here for simplicity)
        self.classifier = pipeline("zero-shot-classification", model=model_id)
        
        self.unsafe_categories = [
            "hate speech", 
            "violence", 
            "illegal acts", 
            "self-harm", 
            "sexual content", 
            "harassment"
        ]

    def validate(self, text: str) -> bool:
        """
        Checks if the text triggers any unsafe categories above the threshold.
        Returns True if safe, False if unsafe.
        """
        # A lightweight zero-shot classification
        result = self.classifier(text, self.unsafe_categories, multi_label=True)
        
        for score in result['scores']:
            if score > self.threshold:
                return False
        return True
