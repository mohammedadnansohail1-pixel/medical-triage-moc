"""
Skin Lesion Classifier using Swin Transformer.

Classifies skin images into 8 lesion types with 4-tier risk stratification
aligned with NICE NG12 clinical pathways.

TIERS (aligned with NICE suspected cancer pathways):
- ROUTINE_MONITORING: Clearly benign, self-monitor for changes
- CONSIDER_EVALUATION: Low concern, consider GP review if worried  
- ROUTINE_REFERRAL: Non-urgent dermatology appointment recommended
- URGENT_REFERRAL: 2-week suspected cancer pathway
"""

import torch
from PIL import Image
from typing import Dict, List, Optional
from enum import Enum
import structlog
from transformers import AutoModelForImageClassification, AutoImageProcessor

logger = structlog.get_logger(__name__)

MODEL_ID = "NeuronZero/SkinCancerClassifier"


class RiskTier(str, Enum):
    """Risk tiers aligned with NICE NG12 pathways."""
    ROUTINE_MONITORING = "routine_monitoring"
    CONSIDER_EVALUATION = "consider_evaluation"
    ROUTINE_REFERRAL = "routine_referral"
    URGENT_REFERRAL = "urgent_referral"


# Tier display information
TIER_INFO = {
    RiskTier.ROUTINE_MONITORING: {
        "display_name": "Routine Monitoring",
        "color": "green",
        "timeframe": "Self-monitor",
        "message": (
            "This appears to be a benign (non-cancerous) skin lesion. No immediate medical "
            "attention is needed, but monitor for any changes in size, shape, color, or symptoms. "
            "If you notice changes, consult your doctor."
        ),
        "action": "Continue regular skin self-checks",
    },
    RiskTier.CONSIDER_EVALUATION: {
        "display_name": "Consider Evaluation", 
        "color": "yellow",
        "timeframe": "When convenient",
        "message": (
            "This lesion has some features worth noting. While likely benign, consider having "
            "it evaluated by your GP or a dermatologist at your next convenient opportunity, "
            "especially if you have concerns or risk factors for skin cancer."
        ),
        "action": "Consider GP visit if concerned",
    },
    RiskTier.ROUTINE_REFERRAL: {
        "display_name": "Routine Referral",
        "color": "orange",
        "timeframe": "Within a few weeks",
        "message": (
            "This lesion has features that should be evaluated by a dermatologist. While not "
            "an emergency, we recommend scheduling an appointment within the next few weeks "
            "for professional assessment and possible biopsy."
        ),
        "action": "Schedule dermatology appointment",
    },
    RiskTier.URGENT_REFERRAL: {
        "display_name": "Urgent Referral",
        "color": "red",
        "timeframe": "Within 2 weeks",
        "message": (
            "This lesion has features concerning for possible skin cancer. Please see a "
            "dermatologist as soon as possible, ideally within 2 weeks. Early evaluation "
            "and treatment significantly improve outcomes. Contact your GP for an urgent referral."
        ),
        "action": "Seek urgent dermatology referral",
    },
}


# Class labels with clinical information
CLASS_INFO = {
    "AK": {
        "full_name": "Actinic Keratosis",
        "description": "Pre-cancerous scaly patches caused by sun damage",
        "risk_level": "medium",
        "icd10": "L57.0",
        "is_cancer": False,
        "is_precancer": True,
        "transformation_risk": "0.1% per year",
    },
    "BCC": {
        "full_name": "Basal Cell Carcinoma",
        "description": "Most common type of skin cancer, slow-growing, rarely spreads",
        "risk_level": "high",
        "icd10": "C44.91",
        "is_cancer": True,
        "is_precancer": False,
        "clinical_note": "Typically requires routine (non-urgent) dermatology referral per NICE guidelines",
    },
    "BKL": {
        "full_name": "Benign Keratosis",
        "description": "Non-cancerous skin growths (seborrheic keratosis)",
        "risk_level": "low",
        "icd10": "L82.1",
        "is_cancer": False,
        "is_precancer": False,
    },
    "DF": {
        "full_name": "Dermatofibroma",
        "description": "Benign fibrous skin nodule",
        "risk_level": "low",
        "icd10": "D23.9",
        "is_cancer": False,
        "is_precancer": False,
    },
    "MEL": {
        "full_name": "Melanoma",
        "description": "Most dangerous type of skin cancer, can spread quickly",
        "risk_level": "high",
        "icd10": "C43.9",
        "is_cancer": True,
        "is_precancer": False,
        "clinical_note": "Requires urgent 2-week suspected cancer pathway referral",
    },
    "NV": {
        "full_name": "Melanocytic Nevus",
        "description": "Common mole, usually benign",
        "risk_level": "low",
        "icd10": "D22.9",
        "is_cancer": False,
        "is_precancer": False,
    },
    "SCC": {
        "full_name": "Squamous Cell Carcinoma",
        "description": "Second most common skin cancer, can spread if untreated",
        "risk_level": "high",
        "icd10": "C44.92",
        "is_cancer": True,
        "is_precancer": False,
        "clinical_note": "Requires urgent 2-week suspected cancer pathway referral",
    },
    "VASC": {
        "full_name": "Vascular Lesion",
        "description": "Blood vessel-related skin marks (angiomas, etc.)",
        "risk_level": "low",
        "icd10": "D18.01",
        "is_cancer": False,
        "is_precancer": False,
    },
}

# Class groupings
URGENT_CANCER_CLASSES = {"MEL", "SCC"}  # 2-week pathway
ROUTINE_CANCER_CLASSES = {"BCC"}  # Routine referral
PRECANCER_CLASSES = {"AK"}
BENIGN_CLASSES = {"BKL", "DF", "NV", "VASC"}
ALL_CANCER_CLASSES = URGENT_CANCER_CLASSES | ROUTINE_CANCER_CLASSES

# Thresholds for tier assignment
THRESHOLDS = {
    "benign_high_confidence": 0.85,    # >85% benign = green tier
    "benign_moderate_confidence": 0.70, # 70-85% benign = yellow tier
    "cancer_low": 0.05,                 # <5% cancer for green tier
    "cancer_moderate": 0.15,            # 5-15% cancer = yellow tier
    "cancer_high": 0.30,                # 15-30% cancer = orange tier
    "melanoma_alert": 0.03,             # >3% MEL alone triggers yellow minimum
}


class SkinLesionClassifier:
    """Classifies skin lesion images with 4-tier risk stratification."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load model and processor from HuggingFace."""
        if self._loaded:
            return

        logger.info("loading_skin_classifier", model=MODEL_ID, device=self.device)

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info("skin_classifier_loaded", device=self.device)

    def _calculate_probabilities(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate probabilities."""
        probs = {
            "urgent_cancer": 0.0,  # MEL + SCC
            "routine_cancer": 0.0,  # BCC
            "all_cancer": 0.0,
            "precancer": 0.0,
            "benign": 0.0,
            "melanoma": 0.0,
        }
        
        for p in predictions:
            code = p["class_code"]
            prob = p["probability"]
            
            if code in URGENT_CANCER_CLASSES:
                probs["urgent_cancer"] += prob
            if code in ROUTINE_CANCER_CLASSES:
                probs["routine_cancer"] += prob
            if code in ALL_CANCER_CLASSES:
                probs["all_cancer"] += prob
            if code in PRECANCER_CLASSES:
                probs["precancer"] += prob
            if code in BENIGN_CLASSES:
                probs["benign"] += prob
            if code == "MEL":
                probs["melanoma"] = prob
                
        return probs

    def _determine_tier(
        self, 
        top_class: str, 
        top_confidence: float,
        probs: Dict[str, float]
    ) -> tuple[RiskTier, List[str]]:
        """
        Determine risk tier based on predictions.
        
        Returns (tier, reasons) tuple.
        """
        reasons = []
        
        # URGENT REFERRAL: MEL/SCC predicted OR high cancer probability
        if top_class in URGENT_CANCER_CLASSES:
            reasons.append(f"Top prediction is {CLASS_INFO[top_class]['full_name']}")
            return RiskTier.URGENT_REFERRAL, reasons
        
        if probs["urgent_cancer"] > THRESHOLDS["cancer_high"]:
            reasons.append(f"High melanoma/SCC probability ({probs['urgent_cancer']:.1%})")
            return RiskTier.URGENT_REFERRAL, reasons
        
        # ROUTINE REFERRAL: BCC predicted OR moderate-high cancer probability
        if top_class in ROUTINE_CANCER_CLASSES:
            reasons.append(f"Top prediction is {CLASS_INFO[top_class]['full_name']}")
            return RiskTier.ROUTINE_REFERRAL, reasons
        
        if probs["all_cancer"] > THRESHOLDS["cancer_high"]:
            reasons.append(f"Elevated cancer probability ({probs['all_cancer']:.1%})")
            return RiskTier.ROUTINE_REFERRAL, reasons
        
        if probs["urgent_cancer"] > THRESHOLDS["cancer_moderate"]:
            reasons.append(f"Moderate melanoma/SCC probability ({probs['urgent_cancer']:.1%})")
            return RiskTier.ROUTINE_REFERRAL, reasons
        
        # CONSIDER EVALUATION: AK, moderate cancer probability, low confidence benign
        if top_class in PRECANCER_CLASSES:
            reasons.append("Pre-cancerous lesion (Actinic Keratosis) detected")
            return RiskTier.CONSIDER_EVALUATION, reasons
        
        if probs["all_cancer"] > THRESHOLDS["cancer_moderate"]:
            reasons.append(f"Moderate cancer probability ({probs['all_cancer']:.1%})")
            return RiskTier.CONSIDER_EVALUATION, reasons
        
        if probs["melanoma"] > THRESHOLDS["melanoma_alert"]:
            reasons.append(f"Melanoma probability ({probs['melanoma']:.1%}) warrants verification")
            return RiskTier.CONSIDER_EVALUATION, reasons
        
        if probs["all_cancer"] > THRESHOLDS["cancer_low"]:
            reasons.append(f"Low but non-trivial cancer probability ({probs['all_cancer']:.1%})")
            return RiskTier.CONSIDER_EVALUATION, reasons
        
        if top_class in BENIGN_CLASSES and top_confidence < THRESHOLDS["benign_moderate_confidence"]:
            reasons.append(f"Low confidence ({top_confidence:.1%}) in benign classification")
            return RiskTier.CONSIDER_EVALUATION, reasons
        
        if top_class in BENIGN_CLASSES and top_confidence < THRESHOLDS["benign_high_confidence"]:
            reasons.append(f"Moderate confidence ({top_confidence:.1%}) - consider evaluation if concerned")
            return RiskTier.CONSIDER_EVALUATION, reasons
        
        # ROUTINE MONITORING: High-confidence benign
        if top_class in BENIGN_CLASSES and top_confidence >= THRESHOLDS["benign_high_confidence"]:
            if probs["all_cancer"] < THRESHOLDS["cancer_low"]:
                reasons.append(f"High confidence benign lesion ({top_confidence:.1%})")
                return RiskTier.ROUTINE_MONITORING, reasons
        
        # Default to CONSIDER_EVALUATION for safety
        reasons.append("Unable to confidently classify - evaluation recommended")
        return RiskTier.CONSIDER_EVALUATION, reasons

    def predict(self, image: Image.Image) -> Dict:
        """
        Classify a skin lesion image with 4-tier risk stratification.

        Args:
            image: PIL Image of skin lesion

        Returns:
            Dict with prediction results and tier assignment
        """
        if not self._loaded:
            self.load()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess and predict
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        # Build predictions list
        probs_np = probabilities.cpu().numpy()
        id2label = self.model.config.id2label
        sorted_indices = probs_np.argsort()[::-1]
        
        all_predictions = []
        for idx in sorted_indices:
            class_code = id2label[idx]
            prob = float(probs_np[idx])
            info = CLASS_INFO.get(class_code, {})
            all_predictions.append({
                "class_code": class_code,
                "class_name": info.get("full_name", class_code),
                "probability": prob,
                "risk_level": info.get("risk_level", "unknown"),
                "is_cancer": info.get("is_cancer", False),
                "is_precancer": info.get("is_precancer", False),
            })

        # Top prediction
        top_pred = all_predictions[0]
        top_class = top_pred["class_code"]
        confidence = top_pred["probability"]

        # Calculate aggregate probabilities
        probs = self._calculate_probabilities(all_predictions)
        
        # Determine tier
        tier, reasons = self._determine_tier(top_class, confidence, probs)
        tier_info = TIER_INFO[tier]

        result = {
            "prediction": top_class,
            "prediction_label": top_pred["class_name"],
            "confidence": confidence,
            "description": CLASS_INFO.get(top_class, {}).get("description", ""),
            "icd10": CLASS_INFO.get(top_class, {}).get("icd10", ""),
            "all_predictions": all_predictions[:5],
            "probability_summary": {
                "benign": probs["benign"],
                "precancer": probs["precancer"],
                "cancer_total": probs["all_cancer"],
                "melanoma": probs["melanoma"],
            },
            "risk_assessment": {
                "tier": tier.value,
                "tier_display": tier_info["display_name"],
                "color": tier_info["color"],
                "timeframe": tier_info["timeframe"],
                "message": tier_info["message"],
                "action": tier_info["action"],
                "reasons": reasons,
            },
            "disclaimer": (
                "This is an AI-assisted screening tool for informational purposes only. "
                "It is NOT a medical diagnosis. Accuracy varies and false results can occur. "
                "Always consult a qualified healthcare professional for proper evaluation."
            ),
        }

        logger.info(
            "skin_classification_complete",
            prediction=result["prediction"],
            confidence=f"{confidence:.1%}",
            tier=tier.value,
        )

        return result

    def classify_with_context(
        self,
        image: Image.Image,
        symptoms: Optional[List[str]] = None,
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> Dict:
        """
        Classify image with patient context that may adjust tier.
        """
        result = self.predict(image)
        result["context"] = {"symptoms": symptoms or [], "age": age, "sex": sex}
        
        # Age-based tier adjustment
        additional_factors = []
        current_tier = RiskTier(result["risk_assessment"]["tier"])
        
        if age is not None and age > 50:
            additional_factors.append(f"Age ({age}) is a risk factor for skin cancer")
            # Upgrade ROUTINE_MONITORING to CONSIDER_EVALUATION for older patients
            if current_tier == RiskTier.ROUTINE_MONITORING:
                current_tier = RiskTier.CONSIDER_EVALUATION
                result["risk_assessment"]["tier"] = current_tier.value
                tier_info = TIER_INFO[current_tier]
                result["risk_assessment"]["tier_display"] = tier_info["display_name"]
                result["risk_assessment"]["color"] = tier_info["color"]
                result["risk_assessment"]["timeframe"] = tier_info["timeframe"]
                result["risk_assessment"]["message"] = tier_info["message"]
                result["risk_assessment"]["action"] = tier_info["action"]
        
        # Symptom-based adjustment
        if symptoms:
            concerning = ["changing", "growing", "bleeding", "itching", "new", "pain"]
            for symptom in symptoms:
                if any(c in symptom.lower() for c in concerning):
                    additional_factors.append(f"Symptom '{symptom}' warrants attention")
                    if current_tier == RiskTier.ROUTINE_MONITORING:
                        current_tier = RiskTier.CONSIDER_EVALUATION
                        result["risk_assessment"]["tier"] = current_tier.value
                        tier_info = TIER_INFO[current_tier]
                        result["risk_assessment"]["tier_display"] = tier_info["display_name"]
                        result["risk_assessment"]["color"] = tier_info["color"]
                        result["risk_assessment"]["timeframe"] = tier_info["timeframe"]
                        result["risk_assessment"]["message"] = tier_info["message"]
                        result["risk_assessment"]["action"] = tier_info["action"]
                    break
        
        if additional_factors:
            result["risk_assessment"]["reasons"].extend(additional_factors)
        
        return result


# Singleton
_classifier: Optional[SkinLesionClassifier] = None


def get_skin_classifier() -> SkinLesionClassifier:
    """Get or create singleton classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = SkinLesionClassifier()
    return _classifier
