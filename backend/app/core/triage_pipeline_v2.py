"""
Two-Stage Triage Pipeline:
1. SapBERT: Patient text → DDXPlus evidence codes
2. XGBoost: Evidence codes → Specialty (99% accuracy)
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog

from app.core.sapbert_linker import SapBERTLinker, get_sapbert_linker

logger = structlog.get_logger(__name__)


class TriagePipelineV2:
    """Two-stage pipeline: SapBERT linking + XGBoost classification."""

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

        # Load SapBERT and build evidence index
        self.sapbert = get_sapbert_linker()
        self.sapbert.load()
        self.sapbert.build_evidence_index(evidences_path)

        # Load XGBoost model
        with open(model_path, "rb") as f:
            self.xgboost_model = pickle.load(f)

        # Load vocabulary
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        self.code_to_idx = vocab["code_to_idx"]
        self.idx_to_specialty = vocab["idx_to_specialty"]

        self._loaded = True
        logger.info(
            "pipeline_loaded",
            num_evidence_codes=len(self.code_to_idx),
            num_specialties=len(self.idx_to_specialty),
        )

    def unload(self) -> None:
        """Free resources."""
        if self.sapbert:
            self.sapbert.unload()
        self.xgboost_model = None
        self._loaded = False
        logger.info("pipeline_unloaded")

    def _symptoms_to_feature_vector(self, symptoms: List[str], threshold: float = 0.4) -> Tuple[np.ndarray, set]:
        """Convert patient symptoms to evidence code feature vector."""
        # Link symptoms to evidence codes via SapBERT
        matches = self.sapbert.link_symptoms(symptoms, top_k=3, threshold=threshold)

        # XGBoost expects 225 features
        features = np.zeros(225, dtype=np.float32)

        matched_codes = set()
        for symptom, code, score in matches:
            if code in self.code_to_idx:
                idx = self.code_to_idx[code]
                features[idx] = 1.0
                matched_codes.add(code)

        logger.info(
            "symptoms_vectorized",
            input_symptoms=len(symptoms),
            matched_codes=len(matched_codes),
        )

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
            Dict with specialty, confidence, matched codes, reasoning
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded")

        # Stage 1: SapBERT linking
        features, matched_codes = self._symptoms_to_feature_vector(symptoms, threshold)

        if not matched_codes:
            return {
                "specialty": "general_medicine",
                "confidence": 0.3,
                "matched_codes": [],
                "reasoning": ["No symptoms matched to known evidence codes"],
            }

        # Stage 2: XGBoost prediction
        features_2d = features.reshape(1, -1)
        proba = self.xgboost_model.predict_proba(features_2d)[0]

        top_idx = np.argmax(proba)
        specialty = self.idx_to_specialty[top_idx]
        confidence = float(proba[top_idx])

        # Get top 3 for reasoning
        top_3_idx = np.argsort(proba)[-3:][::-1]
        reasoning = [
            f"{self.idx_to_specialty[i]}: {proba[i]:.1%}"
            for i in top_3_idx
        ]

        return {
            "specialty": specialty,
            "confidence": confidence,
            "matched_codes": list(matched_codes),
            "reasoning": reasoning,
        }


_pipeline: Optional[TriagePipelineV2] = None


def get_triage_pipeline() -> TriagePipelineV2:
    """Get or create singleton pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = TriagePipelineV2()
    return _pipeline
