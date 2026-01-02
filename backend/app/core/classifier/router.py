"""
Classifier-based router for symptom-to-specialty routing.
Uses trained XGBoost model on evidence codes.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ClassifierResult:
    """Result from classifier routing."""
    
    primary_specialty: str
    confidence: float
    all_probabilities: Dict[str, float]


class ClassifierRouter:
    """Route symptoms to specialties using trained XGBoost classifier."""
    
    def __init__(self, model_dir: str = "data/classifier"):
        """Load trained model and vocabulary."""
        self.model_dir = Path(model_dir)
        self._model = None
        self._vocab = None
        self._loaded = False
    
    def _load(self) -> None:
        """Lazy load model and vocabulary."""
        if self._loaded:
            return
        
        model_path = self.model_dir / "model.pkl"
        vocab_path = self.model_dir / "vocabulary.pkl"
        
        if not model_path.exists() or not vocab_path.exists():
            raise FileNotFoundError(
                f"Model or vocabulary not found in {self.model_dir}. "
                "Run train.py first."
            )
        
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        
        with open(vocab_path, "rb") as f:
            self._vocab = pickle.load(f)
        
        self._loaded = True
        logger.info(
            "classifier_loaded",
            n_features=len(self._vocab["code_to_idx"]) + 2,
            n_classes=len(self._vocab["idx_to_specialty"]),
        )
    
    def route(
        self,
        symptoms: List[str],
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> ClassifierResult:
        """
        Route symptoms to specialty using classifier.
        
        Args:
            symptoms: List of symptom names (will be mapped to evidence codes)
            age: Patient age
            sex: Patient sex (male/female)
            
        Returns:
            ClassifierResult with specialty and confidence
        """
        self._load()
        
        # Map symptoms to evidence codes
        # Note: Our symptoms are already extracted as canonical names
        # We need to create a feature vector
        code_to_idx = self._vocab["code_to_idx"]
        idx_to_specialty = self._vocab["idx_to_specialty"]
        
        # Create feature vector
        feature_vec = np.zeros(len(code_to_idx), dtype=np.float32)
        
        # Map symptom names to likely evidence codes
        symptom_to_codes = self._get_symptom_code_mapping()
        
        for symptom in symptoms:
            symptom_lower = symptom.lower().replace("_", " ")
            for symptom_key, codes in symptom_to_codes.items():
                if symptom_key in symptom_lower or symptom_lower in symptom_key:
                    for code in codes:
                        if code in code_to_idx:
                            feature_vec[code_to_idx[code]] = 1.0
        
        # Add age and sex
        age_norm = (age / 100.0) if age else 0.5
        sex_val = 1.0 if sex == "male" else 0.0 if sex == "female" else 0.5
        
        feature_vec = np.append(feature_vec, [age_norm, sex_val])
        feature_vec = feature_vec.reshape(1, -1)
        
        # Get prediction probabilities
        probas = self._model.predict_proba(feature_vec)[0]
        pred_idx = np.argmax(probas)
        
        all_probs = {
            idx_to_specialty[i]: float(probas[i])
            for i in range(len(probas))
        }
        
        result = ClassifierResult(
            primary_specialty=idx_to_specialty[pred_idx],
            confidence=float(probas[pred_idx]),
            all_probabilities=all_probs,
        )
        
        logger.info(
            "classifier_routing_complete",
            primary_specialty=result.primary_specialty,
            confidence=result.confidence,
        )
        
        return result
    
    def route_from_evidences(
        self,
        evidence_codes: List[str],
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> ClassifierResult:
        """
        Route directly from DDXPlus evidence codes.
        
        Args:
            evidence_codes: List of evidence codes (E_XX format)
            age: Patient age
            sex: Patient sex
            
        Returns:
            ClassifierResult
        """
        self._load()
        
        code_to_idx = self._vocab["code_to_idx"]
        idx_to_specialty = self._vocab["idx_to_specialty"]
        
        # Create feature vector from evidence codes
        feature_vec = np.zeros(len(code_to_idx), dtype=np.float32)
        
        for code in evidence_codes:
            base_code = code.split("_@_")[0]
            if base_code in code_to_idx:
                feature_vec[code_to_idx[base_code]] = 1.0
        
        # Add age and sex
        age_norm = (age / 100.0) if age else 0.5
        sex_val = 1.0 if sex == "male" else 0.0 if sex == "female" else 0.5
        
        feature_vec = np.append(feature_vec, [age_norm, sex_val])
        feature_vec = feature_vec.reshape(1, -1)
        
        # Get prediction
        probas = self._model.predict_proba(feature_vec)[0]
        pred_idx = np.argmax(probas)
        
        all_probs = {
            idx_to_specialty[i]: float(probas[i])
            for i in range(len(probas))
        }
        
        return ClassifierResult(
            primary_specialty=idx_to_specialty[pred_idx],
            confidence=float(probas[pred_idx]),
            all_probabilities=all_probs,
        )
    
    def _get_symptom_code_mapping(self) -> Dict[str, List[str]]:
        """Map symptom names to likely evidence codes."""
        # This is a simplified mapping - in production you'd use the full
        # release_evidences.json mapping
        return {
            "chest pain": ["E_53", "E_54"],
            "shortness of breath": ["E_50", "E_70"],
            "cough": ["E_78", "E_77"],
            "fever": ["E_91", "E_98"],
            "headache": ["E_8", "E_136"],
            "nausea": ["E_140"],
            "vomiting": ["E_204"],
            "dizziness": ["E_217", "E_4"],
            "fatigue": ["E_167", "E_6"],
            "sweating": ["E_173"],
            "palpitations": ["E_49"],
            "abdominal pain": ["E_0", "E_55"],
            "back pain": ["E_17"],
            "joint pain": ["E_16"],
            "rash": ["E_27"],
            "numbness": ["E_32"],
            "confusion": ["E_36"],
            "seizure": ["E_35"],
            "loss of consciousness": ["E_159"],
            "heartburn": ["E_201", "E_45"],
            "sore throat": ["E_97", "E_19"],
            "runny nose": ["E_20"],
            "wheezing": ["E_48"],
        }


# Singleton instance
classifier_router = ClassifierRouter()
