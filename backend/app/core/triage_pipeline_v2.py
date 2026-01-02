"""
Two-Stage Triage Pipeline:
1. Symptom Normalization: Patient language → Medical terms (rule-based)
2. Emergency Rules: Check for safety-critical symptoms (100% reliable)
3. SapBERT: Normalized text → DDXPlus evidence codes
4. XGBoost: Evidence codes → Specialty
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

from app.core.sapbert_linker import SapBERTLinker, get_sapbert_linker
from app.core.emergency_detector import check_emergency_keywords
from app.core.symptom_normalizer import normalize_symptoms

logger = structlog.get_logger(__name__)


class TriagePipelineV2:
    """Four-stage pipeline: Normalize + Emergency rules + SapBERT + XGBoost."""

    def __init__(self) -> None:
        self.sapbert: Optional[SapBERTLinker] = None
        self.xgboost_model = None
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_specialty: Dict[int, str] = {}
        self._loaded = False
        self.emergency_idx: Optional[int] = None

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

    def _symptoms_to_feature_vector(self, symptoms: List[str], threshold: float = 0.3) -> Tuple[np.ndarray, set]:
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
        threshold: float = 0.3,
    ) -> Dict:
        """Predict specialty from patient symptoms."""
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded")

        # Stage 0: Normalize symptoms (patient language → medical terms)
        normalized = normalize_symptoms(symptoms)
        logger.info("symptoms_normalized", original=symptoms, normalized=normalized)

        # Stage 1: Emergency detection (rule-based)
        emergency_result = check_emergency_keywords(symptoms)  # Check original text
        
        if emergency_result["is_emergency"]:
            return {
                "specialty": "emergency",
                "confidence": 1.0,
                "matched_codes": [],
                "reasoning": [f"Emergency detected: {emergency_result['reason']}"],
                "emergency": emergency_result,
                "route": "EMERGENCY_OVERRIDE",
            }

        # Stage 2: SapBERT linking (on normalized text)
        features, matched_codes = self._symptoms_to_feature_vector(normalized, threshold)

        if not matched_codes:
            return {
                "specialty": "general_medicine",
                "confidence": 0.3,
                "matched_codes": [],
                "reasoning": ["No symptoms matched to known evidence codes"],
                "emergency": emergency_result,
                "route": "DEFAULT_FALLBACK",
            }

        # Stage 3: XGBoost prediction
        features_2d = features.reshape(1, -1)
        proba = self.xgboost_model.predict_proba(features_2d)[0]

        # Filter out ML emergency predictions
        top_idx = np.argmax(proba)
        if top_idx == self.emergency_idx:
            proba_filtered = proba.copy()
            proba_filtered[self.emergency_idx] = 0
            top_idx = np.argmax(proba_filtered)
            logger.info("ml_emergency_downgraded", original_conf=float(proba[self.emergency_idx]))

        specialty = self.idx_to_specialty[top_idx]
        confidence = float(proba[top_idx])

        top_3_idx = np.argsort(proba)[-3:][::-1]
        reasoning = [
            f"{self.idx_to_specialty[i]}: {proba[i]:.1%}"
            for i in top_3_idx
            if i != self.emergency_idx
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
