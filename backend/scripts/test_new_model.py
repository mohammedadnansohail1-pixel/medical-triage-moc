#!/usr/bin/env python3
"""Quick test of the new model."""

import pickle
import numpy as np

# Load model
with open("backend/data/classifier/model_sapbert_aligned.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
code_to_idx = data["code_to_idx"]
idx_to_specialty = data["idx_to_specialty"]

print(f"Model loaded successfully!")
print(f"Vocabulary size: {len(code_to_idx)} codes")
print(f"Specialties: {list(idx_to_specialty.values())}")
print(f"\nMetrics:")
for k, v in data["metrics"].items():
    print(f"  {k}: {v['accuracy']:.1%} acc, {v['macro_f1']:.1%} F1")

# Test prediction
def predict(codes: list) -> dict:
    vec = np.zeros(len(code_to_idx), dtype=np.float32)
    for c in codes:
        if c in code_to_idx:
            vec[code_to_idx[c]] = 1.0
    
    proba = model.predict_proba([vec])[0]
    pred_idx = np.argmax(proba)
    
    return {
        "specialty": idx_to_specialty[pred_idx],
        "confidence": float(proba[pred_idx]),
        "all_probs": {idx_to_specialty[i]: float(p) for i, p in enumerate(proba)}
    }

# Example
test_codes = ["E_15", "E_44", "E_45"]  # chest pain related
result = predict(test_codes)
print(f"\nTest prediction for {test_codes}:")
print(f"  Specialty: {result['specialty']}")
print(f"  Confidence: {result['confidence']:.1%}")
