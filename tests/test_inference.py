import pytest
from core.inference.guardrails import Guardrails

# We mock the pipeline so we don't download bart-large-mnli during tests
def test_guardrails(mocker):
    # Mock the pipeline return value
    mock_pipeline = mocker.patch("core.inference.guardrails.pipeline")
    mock_classifier = mocker.MagicMock()
    
    # Simulate a safe result
    mock_classifier.return_value = {"scores": [0.1, 0.2, 0.05]}
    mock_pipeline.return_value = mock_classifier
    
    g = Guardrails()
    assert g.validate("This is a safe string.") is True
    
    # Simulate an unsafe result (score > threshold 0.8)
    mock_classifier.return_value = {"scores": [0.1, 0.95, 0.05]}
    assert g.validate("This is an unsafe string.") is False
