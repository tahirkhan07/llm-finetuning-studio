import warnings
from transformers import pipeline
from config import ACTIVE_DOMAIN

# Suppress some noisy warnings from the pipeline
warnings.filterwarnings("ignore", category=UserWarning)

print("Loading Guardrails Classifier...")
# We use a fast zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

DOMAIN_TOPICS = {
    "medical": ["medicine", "healthcare", "disease", "treatment", "biology", "anatomy", "symptoms", "health"],
    "coding": ["programming", "software engineering", "coding", "computer science", "technology", "debugging", "python", "code"],
    "law": ["law", "legal", "court", "attorney", "justice", "contract", "legislation", "lawsuit"],
    "finance": ["finance", "investment", "stock market", "economics", "banking", "trading", "accounting", "financial"]
}

def validate_query(query: str, domain: str = ACTIVE_DOMAIN) -> bool:
    """
    Returns True if the query is related to the domain, False otherwise.
    """
    topics = DOMAIN_TOPICS.get(domain, [domain])
    # Include out-of-domain categories so the model has negative targets
    labels = topics + ["other", "unrelated", "chit-chat", "casual conversation", "general knowledge", "cooking", "sports"]
    
    result = classifier(query, candidate_labels=labels)
    
    # Check what the highest scoring label is
    top_label = result["labels"][0]
    
    if top_label in topics:
        return True
    
    # Give it some leeway if it's very close or ambiguous
    # If any valid topic is above 0.25 probability, we allow it to pass to the main model
    for label, score in zip(result["labels"], result["scores"]):
        if label in topics and score > 0.25:
            return True
            
    return False

if __name__ == "__main__":
    print("\n--- Testing Guardrails ---")
    
    test_queries = [
        ("How do I write a for loop in python?", "coding"),
        ("What are the symptoms of the flu?", "coding"),
        ("Explain the legal doctrine of strict liability.", "law"),
        ("Write a function to sort an array", "law"),
        ("Identify the cause of unusual vaginal discharge", "medical"),
        ("How do I fix a segmentation fault?", "medical"),
    ]
    
    for query, domain in test_queries:
        is_valid = validate_query(query, domain)
        status = "PASSED" if is_valid else "BLOCKED"
        print(f"[{status}] (Domain: {domain}) Query: {query}")
