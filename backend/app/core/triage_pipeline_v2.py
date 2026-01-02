"""
Two-Stage Triage Pipeline:
1. Symptom Normalization: Patient language → Medical terms (rule-based)
2. Emergency Rules: Check for safety-critical symptoms (100% reliable)
3. SapBERT: Normalized text → DDXPlus evidence codes
4. XGBoost: Evidence codes → Specialty
5. Specialty Agent: Evidence codes → Differential diagnosis within specialty
6. LLM Explanation: Generate patient-friendly explanation
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from app.core.sapbert_linker import SapBERTLinker, get_sapbert_linker
from app.core.emergency_detector import check_emergency_keywords
from app.core.symptom_normalizer import normalize_symptoms
from app.core.specialty_agent import SpecialtyAgentManager, get_specialty_manager
from app.core.explanation_generator import ExplanationGenerator, get_explanation_generator

logger = structlog.get_logger(__name__)

# Rule-based specialty detection for cases where XGBoost training data is misaligned
# These rules fire BEFORE XGBoost when symptom patterns strongly indicate a specialty
SYMPTOM_SPECIALTY_RULES: Dict[str, Dict] = {
    "dermatology": {
        "keywords": ["rash", "skin rash", "itch", "itchy", "hives", "eczema", "acne", 
                     "psoriasis", "dermatitis", "lesion", "blister", "skin irritation",
                     "bumps on skin", "skin redness", "itchy skin", "skin problem"],
        "evidence_codes": ["E_129", "E_130", "E_132", "E_134", "E_136"],
        "min_keyword_matches": 1,  # Just keywords - dermatology has no good training data
        "require_both": False,
        "confidence": 0.85,
    },
    "gastroenterology": {
        "keywords": ["stomach", "nausea", "vomit", "diarrhea", "constipation",
                     "heartburn", "acid reflux", "bloating", "indigestion", "gastric",
                     "abdominal pain", "belly"],
        "evidence_codes": ["E_98", "E_97", "E_125", "E_173"],
        "min_keyword_matches": 2,  # Need strong keyword signal
        "require_both": True,  # Must have BOTH keywords AND codes
        "confidence": 0.80,
    },
}


def _apply_specialty_rules(
    symptoms: List[str], matched_codes: set
) -> Optional[Tuple[str, float]]:
    """
    Apply rule-based specialty detection BEFORE XGBoost.
    
    Returns (specialty, confidence) or None to fall through to XGBoost.
    This fixes cases where DDXPlus training data doesn't align with
    actual symptom-specialty relationships (e.g., dermatology).
    """
    symptoms_lower = [s.lower() for s in symptoms]
    symptoms_text = " ".join(symptoms_lower)
    
    for specialty, rules in SYMPTOM_SPECIALTY_RULES.items():
        # Count keyword matches (check if keyword appears in any symptom)
        keyword_matches = sum(
            1 for kw in rules["keywords"]
            if kw in symptoms_text or any(kw in s for s in symptoms_lower)
        )
        
        # Count evidence code matches
        code_matches = sum(
            1 for code in rules["evidence_codes"]
            if code in matched_codes
        )
        
        # Check if we meet the threshold
        require_both = rules.get("require_both", False)
        
        if require_both:
            # Must have BOTH sufficient keywords AND at least one code
            matches = keyword_matches >= rules["min_keyword_matches"] and code_matches >= 1
        else:
            # Just need sufficient keywords (for specialties with bad training data)
            matches = keyword_matches >= rules["min_keyword_matches"]
        
        if matches:
            logger.info(
                "specialty_rule_matched",
                specialty=specialty,
                keyword_matches=keyword_matches,
                code_matches=code_matches,
            )
            return (specialty, rules["confidence"])
    
    return None


class TriagePipelineV2:
    """Six-stage pipeline: Normalize + Emergency + SapBERT + XGBoost + SpecialtyAgent + LLM Explanation."""

    def __init__(self) -> None:
        self.sapbert: Optional[SapBERTLinker] = None
        self.xgboost_model = None
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_specialty: Dict[int, str] = {}
        self._loaded = False
        self.emergency_idx: Optional[int] = None
        self.specialty_manager: Optional[SpecialtyAgentManager] = None
        self.explanation_generator: Optional[ExplanationGenerator] = None

    def load(
        self,
        evidences_path: Path,
        model_path: Path,
        vocab_path: Path,
        condition_model_path: Optional[Path] = None,
        enable_explanations: bool = True,
    ) -> None:
        """Load all components."""
        if self._loaded:
            return

        logger.info("pipeline_loading")

        # SapBERT
        self.sapbert = get_sapbert_linker()
        self.sapbert.load()
        self.sapbert.build_evidence_index(evidences_path)

        # XGBoost
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

        # Specialty agents
        if condition_model_path is None:
            condition_model_path = evidences_path.parent / "condition_model.json"
        
        if condition_model_path.exists():
            self.specialty_manager = get_specialty_manager(condition_model_path)
            logger.info("specialty_agents_loaded", path=str(condition_model_path))
        else:
            logger.warning("specialty_agents_not_loaded", path=str(condition_model_path))

        # LLM Explanation generator
        if enable_explanations:
            self.explanation_generator = get_explanation_generator()
            if self.explanation_generator.is_available():
                logger.info("explanation_generator_loaded", model=self.explanation_generator.model)
            else:
                logger.warning("explanation_generator_unavailable", msg="Ollama not running")
                self.explanation_generator = None

        self._loaded = True
        logger.info(
            "pipeline_loaded",
            num_evidence_codes=len(self.code_to_idx),
            num_specialties=len(self.idx_to_specialty),
            explanations_enabled=self.explanation_generator is not None,
        )

    def unload(self) -> None:
        """Free resources."""
        if self.sapbert:
            self.sapbert.unload()
        self.xgboost_model = None
        if self.explanation_generator:
            self.explanation_generator.close()
        self._loaded = False
        logger.info("pipeline_unloaded")

    def _symptoms_to_feature_vector(
        self, symptoms: List[str], threshold: float = 0.3
    ) -> Tuple[np.ndarray, set]:
        """Convert patient symptoms to evidence code feature vector."""
        matches = self.sapbert.link_symptoms(symptoms, top_k=3, threshold=threshold)

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

    def _get_differential_diagnosis(
        self,
        specialty: str,
        matched_codes: set,
        age: Optional[int] = None,
        sex: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """Get differential diagnosis from specialty agent."""
        if not self.specialty_manager:
            return []

        try:
            if specialty not in self.specialty_manager.available_specialties:
                logger.warning("specialty_not_available", specialty=specialty)
                return []

            ddx = self.specialty_manager.diagnose(
                specialty=specialty,
                symptom_codes=list(matched_codes),
                age=age,
                sex=sex,
                top_k=top_k,
            )

            return [
                {"condition": condition, "probability": round(prob, 4)}
                for condition, prob in ddx
            ]
        except Exception as e:
            logger.error("differential_diagnosis_failed", error=str(e))
            return []

    def _generate_explanation(
        self,
        symptoms: List[str],
        specialty: str,
        confidence: float,
        differential_diagnosis: List[Dict],
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> Optional[Dict]:
        """Generate LLM explanation if available."""
        if not self.explanation_generator:
            return None

        try:
            result = self.explanation_generator.generate(
                symptoms=symptoms,
                specialty=specialty,
                confidence=confidence,
                differential_diagnosis=differential_diagnosis,
                age=age,
                sex=sex,
            )
            return {
                "text": result.explanation,
                "urgency": result.urgency,
                "next_steps": result.next_steps,
            }
        except Exception as e:
            logger.error("explanation_generation_failed", error=str(e))
            return None

    def predict(
        self,
        symptoms: List[str],
        age: Optional[int] = None,
        sex: Optional[str] = None,
        threshold: float = 0.3,
        include_ddx: bool = True,
        include_explanation: bool = True,
    ) -> Dict:
        """
        Predict specialty, differential diagnosis, and generate explanation.
        
        Args:
            symptoms: Patient symptom descriptions
            age: Patient age
            sex: Patient sex (male/female)
            threshold: SapBERT similarity threshold
            include_ddx: Whether to include differential diagnosis
            include_explanation: Whether to include LLM explanation
            
        Returns:
            Dict with specialty, confidence, differential_diagnosis, explanation, etc.
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded")

        # Stage 0: Normalize symptoms (patient language → medical terms)
        normalized = normalize_symptoms(symptoms)
        logger.info("symptoms_normalized", original=symptoms, normalized=normalized)

        # Stage 1: Emergency detection (rule-based)
        emergency_result = check_emergency_keywords(symptoms)
        if emergency_result["is_emergency"]:
            explanation = None
            if include_explanation and self.explanation_generator:
                explanation = {
                    "text": f"Your symptoms indicate a medical emergency. Please call 911 or go to the nearest emergency room immediately. Reason: {emergency_result['reason']}",
                    "urgency": "emergency",
                    "next_steps": ["Call 911 immediately", "Go to nearest emergency room"],
                }
            
            return {
                "specialty": "emergency",
                "confidence": 1.0,
                "matched_codes": [],
                "reasoning": [f"Emergency detected: {emergency_result['reason']}"],
                "emergency": emergency_result,
                "route": "EMERGENCY_OVERRIDE",
                "differential_diagnosis": [],
                "explanation": explanation,
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
                "differential_diagnosis": [],
                "explanation": None,
            }

        # Stage 3: XGBoost prediction
        features_2d = features.reshape(1, -1)
        proba = self.xgboost_model.predict_proba(features_2d)[0]

        # Filter out ML emergency predictions (only rule-based emergency allowed)
        top_idx = np.argmax(proba)
        if top_idx == self.emergency_idx:
            proba_filtered = proba.copy()
            proba_filtered[self.emergency_idx] = 0
            top_idx = np.argmax(proba_filtered)
            logger.info(
                "ml_emergency_downgraded", original_conf=float(proba[self.emergency_idx])
            )

        specialty = self.idx_to_specialty[top_idx]
        confidence = float(proba[top_idx])

        top_3_idx = np.argsort(proba)[-3:][::-1]
        reasoning = [
            f"{self.idx_to_specialty[i]}: {proba[i]:.1%}"
            for i in top_3_idx
            if i != self.emergency_idx
        ][:3]

        # Stage 4: Differential diagnosis within specialty
        differential_diagnosis = []
        if include_ddx:
            differential_diagnosis = self._get_differential_diagnosis(
                specialty=specialty,
                matched_codes=matched_codes,
                age=age,
                sex=sex,
                top_k=5,
            )

        # Stage 5: LLM Explanation
        explanation = None
        if include_explanation:
            explanation = self._generate_explanation(
                symptoms=symptoms,
                specialty=specialty,
                confidence=confidence,
                differential_diagnosis=differential_diagnosis,
                age=age,
                sex=sex,
            )

        return {
            "specialty": specialty,
            "confidence": confidence,
            "matched_codes": list(matched_codes),
            "reasoning": reasoning,
            "emergency": emergency_result,
            "route": "ML_CLASSIFICATION",
            "differential_diagnosis": differential_diagnosis,
            "explanation": explanation,
        }


_pipeline: Optional[TriagePipelineV2] = None


def get_triage_pipeline() -> TriagePipelineV2:
    """Get or create singleton pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = TriagePipelineV2()
    return _pipeline
