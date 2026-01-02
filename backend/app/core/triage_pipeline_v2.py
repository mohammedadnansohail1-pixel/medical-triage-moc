"""
Two-Stage Triage Pipeline:
1. Emergency Rules: Check for safety-critical symptoms (100% reliable)
2. SapBERT: Patient text → DDXPlus evidence codes
3. XGBoost: Evidence codes → Specialty

IMPORTANT: Only rule-based emergency detection routes to emergency.
ML predictions of "emergency" are downgraded to the next best specialty.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

from app.core.sapbert_linker import SapBERTLinker, get_sapbert_linker
from app.core.emergency_detector import check_emergency_keywords

logger = structlog.get_logger(__name__)


class TriagePipelineV2:
    """Three-stage pipeline: Emergency rules + SapBERT linking + XGBoost classification."""

    def __init__(self) -> None:
        self.sapbert: Optional[SapBERTLinker] = None
        self.xgboost_model = None
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_specialty: Dict[int, str] = {}
        self._loaded = False

    def load(
        self,
        evidences_path: Path,
        model_path: Path,
        vocab_path: Path,
    ) -> None:
        """Load all components."""
        if self._loaded:
            return

        logger.info("pipeline_loading")

        self.sapbert = get_sapbert_linker()
        self.sapbert.load()
        self.sapbert.build_evidence_index(evidences_path)

        with open(model_path, "rb") as f:
            self.xgboost_model = pickle.load(f)

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        self.code_to_idx = vocab["code_to_idx"]
        self.idx_to_specialty = vocab["idx_to_specialty"]
        
        # Find emergency index for filtering
        self.emergency_idx = None
        for idx, name in self.idx_to_specialty.items():
            if name == "emergency":
                self.emergency_idx = idx
                break

        self._loaded = True
        logger.info("pipeline_loaded", num_evidence_codes=len(self.code_to_idx), num_specialties=len(self.idx_to_specialty))

    def unload(self) -> None:
        """Free resources."""
        if self.sapbert:
            self.sapbert.unload()
        self.xgboost_model = None
        self._loaded = False
        logger.info("pipeline_unloaded")

    def _symptoms_to_feature_vector(self, symptoms: List[str], threshold: float = 0.4) -> Tuple[np.ndarray, set]:
        """Convert patient symptoms to evidence code feature vector."""
        matches = self.sapbert.link_symptoms(symptoms, top_k=3, threshold=threshold)

        features = np.zeros(225, dtype=np.float32)

        matched_codes = set()
        for symptom, code, score in matches:
            if code in self.code_to_idx:
                idx = self.code_to_idx[code]
                features[idx] = 1.0
                matched_codes.add(code)

        logger.info("symptoms_vectorized", input_symptoms=len(symptoms), matched_codes=len(matched_codes))

        return features, matched_codes

    def predict(
        self,
        symptoms: List[str],
        threshold: float = 0.4,
    ) -> Dict:
        """
        Predict specialty from patient symptoms.

        Args:
            symptoms: List of symptom strings (patient language)
            threshold: SapBERT similarity threshold

        Returns:
            Dict with specialty, confidence, matched codes, reasoning, emergency info
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded")

        # Stage 0: Emergency detection (rule-based, always first)
        emergency_result = check_emergency_keywords(symptoms)
        
        if emergency_result["is_emergency"]:
            return {
                "specialty": "emergency",
                "confidence": 1.0,
                "matched_codes": [],
                "reasoning": [f"Emergency detected: {emergency_result['reason']}"],
                "emergency": emergency_result,
                "route": "EMERGENCY_OVERRIDE",
            }

        # Stage 1: SapBERT linking
        features, matched_codes = self._symptoms_to_feature_vector(symptoms, threshold)

        if not matched_codes:
            return {
                "specialty": "general_medicine",
                "confidence": 0.3,
                "matched_codes": [],
                "reasoning": ["No symptoms matched to known evidence codes"],
                "emergency": emergency_result,
                "route": "DEFAULT_FALLBACK",
            }

        # Stage 2: XGBoost prediction
        features_2d = features.reshape(1, -1)
        proba = self.xgboost_model.predict_proba(features_2d)[0]

        # IMPORTANT: If ML predicts emergency but rules didn't, use 2nd best
        # Only rule-based detection should route to emergency
        top_idx = np.argmax(proba)
        
        if top_idx == self.emergency_idx:
            # Zero out emergency and get next best
            proba_filtered = proba.copy()
            proba_filtered[self.emergency_idx] = 0
            top_idx = np.argmax(proba_filtered)
            logger.info("ml_emergency_downgraded", original_conf=float(proba[self.emergency_idx]))

        specialty = self.idx_to_specialty[top_idx]
        confidence = float(proba[top_idx])

        # Get top 3 for reasoning (excluding emergency if it was filtered)
        top_3_idx = np.argsort(proba)[-3:][::-1]
        reasoning = [
            f"{self.idx_to_specialty[i]}: {proba[i]:.1%}"
            for i in top_3_idx
            if i != self.emergency_idx or emergency_result["is_emergency"]
        ][:3]

        return {
            "specialty": specialty,
            "confidence": confidence,
            "matched_codes": list(matched_codes),
            "reasoning": reasoning,
            "emergency": emergency_result,
            "route": "ML_CLASSIFICATION",
        }


_pipeline: Optional[TriagePipelineV2] = None


def get_triage_pipeline() -> TriagePipelineV2:
    """Get or create singleton pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = TriagePipelineV2()
    return _pipeline
