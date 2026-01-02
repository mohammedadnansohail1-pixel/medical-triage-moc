#!/usr/bin/env python3
"""Classifier using the retrained SapBERT-aligned model."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class SpecialtyClassifierV2:
    """Specialty classifier using SapBERT-aligned XGBoost model."""
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "data/classifier/model_sapbert_aligned.pkl"
        
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.code_to_idx = data["code_to_idx"]
        self.idx_to_specialty = data["idx_to_specialty"]
        self.specialty_to_idx = data["specialty_to_idx"]
        self.metrics = data["metrics"]
    
    def predict(self, evidence_codes: List[str]) -> Dict:
        """
        Predict specialty from evidence codes.
        
        Args:
            evidence_codes: List of evidence codes (e.g., ["E_15", "E_44"])
        
        Returns:
            Dict with specialty, confidence, and all probabilities
        """
        # Convert codes to feature vector
        vec = np.zeros(len(self.code_to_idx), dtype=np.float32)
        matched_codes = []
        
        for code in evidence_codes:
            if code in self.code_to_idx:
                vec[self.code_to_idx[code]] = 1.0
                matched_codes.append(code)
        
        # Predict
        proba = self.model.predict_proba([vec])[0]
        pred_idx = int(np.argmax(proba))
        
        # Build result
        all_probs = {
            self.idx_to_specialty[i]: float(p) 
            for i, p in enumerate(proba)
        }
        
        # Sort by probability
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "specialty": self.idx_to_specialty[pred_idx],
            "confidence": float(proba[pred_idx]),
            "all_probabilities": dict(sorted_probs),
            "matched_codes": matched_codes,
            "top_3": [{"specialty": s, "confidence": p} for s, p in sorted_probs[:3]],
        }
    
    def predict_batch(self, batch_codes: List[List[str]]) -> List[Dict]:
        """Predict for multiple samples."""
        return [self.predict(codes) for codes in batch_codes]
    
    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            "vocabulary_size": len(self.code_to_idx),
            "specialties": list(self.idx_to_specialty.values()),
            "metrics": self.metrics,
        }


# Singleton instance
_classifier = None

def get_classifier() -> SpecialtyClassifierV2:
    """Get or create classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = SpecialtyClassifierV2()
    return _classifier
