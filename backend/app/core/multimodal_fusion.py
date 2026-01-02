"""
Multimodal Fusion - Decision-level fusion for text + image analysis.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class RiskLevel(IntEnum):
    ROUTINE_MONITORING = 0
    CONSIDER_EVALUATION = 1
    ROUTINE_REFERRAL = 2
    URGENT_REFERRAL = 3
    EMERGENCY = 4


TIER_TO_RISK: Dict[str, RiskLevel] = {
    "routine_monitoring": RiskLevel.ROUTINE_MONITORING,
    "consider_evaluation": RiskLevel.CONSIDER_EVALUATION,
    "routine_referral": RiskLevel.ROUTINE_REFERRAL,
    "urgent_referral": RiskLevel.URGENT_REFERRAL,
    "emergency": RiskLevel.EMERGENCY,
}

URGENCY_TO_RISK: Dict[str, RiskLevel] = {
    "routine": RiskLevel.ROUTINE_MONITORING,
    "non-urgent": RiskLevel.CONSIDER_EVALUATION,
    "soon": RiskLevel.ROUTINE_REFERRAL,
    "urgent": RiskLevel.URGENT_REFERRAL,
    "emergency": RiskLevel.EMERGENCY,
}

RISK_TO_TIER: Dict[RiskLevel, str] = {v: k for k, v in TIER_TO_RISK.items()}

RISK_TO_URGENCY: Dict[RiskLevel, str] = {
    RiskLevel.ROUTINE_MONITORING: "routine",
    RiskLevel.CONSIDER_EVALUATION: "routine",
    RiskLevel.ROUTINE_REFERRAL: "soon",
    RiskLevel.URGENT_REFERRAL: "urgent",
    RiskLevel.EMERGENCY: "emergency",
}

RISK_TO_ACTION: Dict[RiskLevel, str] = {
    RiskLevel.ROUTINE_MONITORING: "Self-monitor; routine skin checks recommended",
    RiskLevel.CONSIDER_EVALUATION: "Consider GP or dermatology appointment",
    RiskLevel.ROUTINE_REFERRAL: "Schedule dermatology appointment within weeks",
    RiskLevel.URGENT_REFERRAL: "See dermatologist within 2 weeks",
    RiskLevel.EMERGENCY: "Seek immediate medical attention",
}


@dataclass
class TextAnalysis:
    specialty: str
    confidence: float
    urgency: str
    differential_diagnosis: List[Dict]
    matched_codes: List[str]


@dataclass
class ImageAnalysis:
    prediction: str
    prediction_label: str
    confidence: float
    tier: str
    tier_display: str
    cancer_probability: float
    melanoma_probability: float
    reasons: List[str]
    message: str


@dataclass
class CombinedAssessment:
    final_risk_tier: str
    final_risk_level: RiskLevel
    final_action: str
    final_urgency: str
    agreement_level: str
    reasoning: str
    text_suggests: Optional[str] = None
    image_suggests: Optional[str] = None
    resolution_rationale: Optional[str] = None


class MultimodalFusion:
    def _text_to_risk(self, text: TextAnalysis) -> RiskLevel:
        return URGENCY_TO_RISK.get(text.urgency, RiskLevel.CONSIDER_EVALUATION)

    def fuse(
        self,
        text_analysis: Optional[TextAnalysis],
        image_analysis: Optional[ImageAnalysis],
    ) -> CombinedAssessment:
        if text_analysis is None and image_analysis is None:
            return CombinedAssessment(
                final_risk_tier="consider_evaluation",
                final_risk_level=RiskLevel.CONSIDER_EVALUATION,
                final_action=RISK_TO_ACTION[RiskLevel.CONSIDER_EVALUATION],
                final_urgency="routine",
                agreement_level="N/A",
                reasoning="Insufficient data for assessment",
            )

        if image_analysis is None:
            text_risk = self._text_to_risk(text_analysis)
            return CombinedAssessment(
                final_risk_tier=RISK_TO_TIER.get(text_risk, "consider_evaluation"),
                final_risk_level=text_risk,
                final_action=RISK_TO_ACTION[text_risk],
                final_urgency=text_analysis.urgency,
                agreement_level="N/A",
                reasoning="Text-only assessment (no image provided)",
            )

        if text_analysis is None:
            image_risk = TIER_TO_RISK.get(image_analysis.tier, RiskLevel.CONSIDER_EVALUATION)
            return CombinedAssessment(
                final_risk_tier=image_analysis.tier,
                final_risk_level=image_risk,
                final_action=RISK_TO_ACTION[image_risk],
                final_urgency=RISK_TO_URGENCY[image_risk],
                agreement_level="N/A",
                reasoning="Image-only assessment",
            )

        text_risk = self._text_to_risk(text_analysis)
        image_risk = TIER_TO_RISK.get(image_analysis.tier, RiskLevel.CONSIDER_EVALUATION)

        risk_diff = abs(int(text_risk) - int(image_risk))
        if risk_diff == 0:
            agreement_level = "strong"
        elif risk_diff == 1:
            agreement_level = "moderate"
        else:
            agreement_level = "conflict"

        final_risk = RiskLevel(max(int(text_risk), int(image_risk)))

        reasoning, resolution = self._build_reasoning(
            text_analysis, image_analysis, text_risk, image_risk, final_risk, agreement_level
        )

        text_suggests = f"{text_analysis.urgency} ({RISK_TO_TIER[text_risk]})"
        image_suggests = f"{image_analysis.tier_display} ({image_analysis.tier})"

        logger.info(
            "multimodal_fusion_complete",
            text_risk=text_risk.name,
            image_risk=image_risk.name,
            final_risk=final_risk.name,
            agreement=agreement_level,
        )

        return CombinedAssessment(
            final_risk_tier=RISK_TO_TIER[final_risk],
            final_risk_level=final_risk,
            final_action=RISK_TO_ACTION[final_risk],
            final_urgency=RISK_TO_URGENCY[final_risk],
            agreement_level=agreement_level,
            reasoning=reasoning,
            text_suggests=text_suggests,
            image_suggests=image_suggests,
            resolution_rationale=resolution if agreement_level == "conflict" else None,
        )

    def _build_reasoning(
        self,
        text: TextAnalysis,
        image: ImageAnalysis,
        text_risk: RiskLevel,
        image_risk: RiskLevel,
        final_risk: RiskLevel,
        agreement: str,
    ) -> Tuple[str, Optional[str]]:
        if agreement == "strong":
            return (
                f"Text and image analysis agree: {RISK_TO_TIER[final_risk].replace('_', ' ')}. "
                f"Image shows {image.prediction_label} with {image.confidence:.0%} confidence.",
                None,
            )

        if agreement == "moderate":
            return (
                f"Text suggests {RISK_TO_TIER[text_risk].replace('_', ' ')}, "
                f"image suggests {RISK_TO_TIER[image_risk].replace('_', ' ')}. "
                f"Taking more conservative approach: {RISK_TO_TIER[final_risk].replace('_', ' ')}.",
                None,
            )

        if image_risk > text_risk:
            resolution = (
                f"Image detected features concerning for {image.prediction_label} "
                f"(cancer prob: {image.cancer_probability:.0%}) despite lower text urgency. "
                "Visual features take precedence for skin lesions."
            )
        else:
            resolution = (
                f"Text symptoms indicate higher urgency ({text.urgency}) "
                f"than image classification ({image.tier_display}). "
                "Clinical symptoms may indicate systemic involvement."
            )

        reasoning = (
            f"CONFLICT: Text suggests {RISK_TO_TIER[text_risk].replace('_', ' ')}, "
            f"but image suggests {RISK_TO_TIER[image_risk].replace('_', ' ')}. "
            f"Conservative approach taken: {RISK_TO_TIER[final_risk].replace('_', ' ')}."
        )

        return reasoning, resolution


_fusion: Optional[MultimodalFusion] = None


def get_multimodal_fusion() -> MultimodalFusion:
    global _fusion
    if _fusion is None:
        _fusion = MultimodalFusion()
    return _fusion
