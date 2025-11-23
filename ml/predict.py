# ml/predict.py

from typing import Dict

def predict_food(image_bytes: bytes) -> Dict[str, float]:
    """
    Dummy predictor for now.
    Later: load real model and do actual prediction.
    """
    # TODO: Use actual ML model here.
    # For now, always return 'chapati' with high confidence.
    return {
        "label": "chapati",
        "confidence": 0.95
    }