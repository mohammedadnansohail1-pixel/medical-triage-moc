"""
Triage API Endpoint - Multimodal support.
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import structlog

from app.core.triage_pipeline_v2 import TriagePipelineV2, get_triage_pipeline
from app.core.image_validator import validate_image
from app.core.multimodal_fusion import (
    get_multimodal_fusion,
    TextAnalysis,
    ImageAnalysis,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["triage"])


class TriageRequest(BaseModel):
    symptoms: List[str] = Field(..., min_length=1)
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, pattern="^(male|female)$")
    include_explanation: bool = Field(True)
    image_base64: Optional[str] = Field(None)


class DifferentialDiagnosis(BaseModel):
    condition: str
    probability: float
    rank: int


class Explanation(BaseModel):
    text: str
    urgency: str
    next_steps: List[str]


class ImageAnalysisResponse(BaseModel):
    prediction: str
    prediction_label: str
    confidence: float
    tier: str
    tier_display: str
    cancer_probability: float
    melanoma_probability: float
    reasons: List[str]
    message: str
    validation_warnings: List[str] = []


class CombinedAssessmentResponse(BaseModel):
    final_risk_tier: str
    final_action: str
    final_urgency: str
    agreement_level: str
    reasoning: str
    text_suggests: Optional[str] = None
    image_suggests: Optional[str] = None
    resolution_rationale: Optional[str] = None


class TriageResponse(BaseModel):
    specialty: str
    confidence: float
    differential_diagnosis: List[DifferentialDiagnosis]
    explanation: Optional[Explanation] = None
    route: Optional[str] = None
    modalities_used: List[str] = ["text"]
    image_analysis: Optional[ImageAnalysisResponse] = None
    combined_assessment: Optional[CombinedAssessmentResponse] = None
    warnings: List[str] = []


_pipeline: Optional[TriagePipelineV2] = None
_skin_classifier = None


def get_data_paths():
    if Path("/app/data/ddxplus/release_evidences.json").exists():
        base = Path("/app")
    else:
        base = Path(__file__).parent.parent.parent.parent
    return {
        "evidences": base / "data" / "ddxplus" / "release_evidences.json",
        "model": Path(__file__).parent.parent.parent / "data" / "classifier" / "model.pkl",
        "vocab": Path(__file__).parent.parent.parent / "data" / "classifier" / "vocabulary.pkl",
    }


def get_pipeline() -> TriagePipelineV2:
    global _pipeline
    if _pipeline is None:
        paths = get_data_paths()
        _pipeline = get_triage_pipeline()
        enable_llm = os.environ.get("ENABLE_LLM", "true").lower() == "true"
        _pipeline.load(
            evidences_path=paths["evidences"],
            model_path=paths["model"],
            vocab_path=paths["vocab"],
            enable_explanations=enable_llm,
        )
    return _pipeline


def get_skin_classifier():
    global _skin_classifier
    if _skin_classifier is None:
        from app.core.skin_classifier import get_skin_classifier as _get_skin
        _skin_classifier = _get_skin()
    return _skin_classifier


def run_image_analysis(image_base64, symptoms, age, sex):
    warnings = []
    validation = validate_image(image_base64)
    if not validation.is_valid:
        return None, validation.errors
    warnings.extend(validation.warnings)
    classifier = get_skin_classifier()
    result = classifier.classify_with_context(
        image=validation.image,
        symptoms=symptoms,
        age=age,
        sex=sex,
    )
    return ImageAnalysisResponse(
        prediction=result["prediction"],
        prediction_label=result["prediction_label"],
        confidence=result["confidence"],
        tier=result["risk_assessment"]["tier"],
        tier_display=result["risk_assessment"]["tier_display"],
        cancer_probability=result["probability_summary"]["cancer_total"],
        melanoma_probability=result["probability_summary"]["melanoma"],
        reasons=result["risk_assessment"]["reasons"],
        message=result["risk_assessment"]["message"],
        validation_warnings=warnings,
    ), warnings


@router.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest) -> TriageResponse:
    try:
        pipeline = get_pipeline()
        result = pipeline.predict(
            symptoms=request.symptoms,
            age=request.age,
            sex=request.sex,
            include_explanation=request.include_explanation,
        )

        ddx = [
            DifferentialDiagnosis(
                condition=d["condition"],
                probability=d["probability"],
                rank=i + 1
            )
            for i, d in enumerate(result.get("differential_diagnosis", []))
        ]

        explanation = None
        if result.get("explanation"):
            exp = result["explanation"]
            explanation = Explanation(
                text=exp["text"],
                urgency=exp["urgency"],
                next_steps=exp["next_steps"]
            )

        modalities = ["text"]
        warnings = []
        image_analysis_resp = None
        combined_resp = None

        is_dermatology = result["specialty"] == "dermatology"
        has_image = request.image_base64 is not None

        if has_image and is_dermatology:
            image_result, img_warnings = run_image_analysis(
                request.image_base64,
                request.symptoms,
                request.age,
                request.sex
            )
            if image_result:
                modalities.append("image")
                image_analysis_resp = image_result
                warnings.extend(img_warnings)

                text_analysis = TextAnalysis(
                    specialty=result["specialty"],
                    confidence=result["confidence"],
                    urgency=explanation.urgency if explanation else "routine",
                    differential_diagnosis=result.get("differential_diagnosis", []),
                    matched_codes=result.get("matched_codes", []),
                )
                img_analysis = ImageAnalysis(
                    prediction=image_result.prediction,
                    prediction_label=image_result.prediction_label,
                    confidence=image_result.confidence,
                    tier=image_result.tier,
                    tier_display=image_result.tier_display,
                    cancer_probability=image_result.cancer_probability,
                    melanoma_probability=image_result.melanoma_probability,
                    reasons=image_result.reasons,
                    message=image_result.message,
                )
                fusion = get_multimodal_fusion()
                combined = fusion.fuse(text_analysis, img_analysis)
                combined_resp = CombinedAssessmentResponse(
                    final_risk_tier=combined.final_risk_tier,
                    final_action=combined.final_action,
                    final_urgency=combined.final_urgency,
                    agreement_level=combined.agreement_level,
                    reasoning=combined.reasoning,
                    text_suggests=combined.text_suggests,
                    image_suggests=combined.image_suggests,
                    resolution_rationale=combined.resolution_rationale,
                )
                if combined.agreement_level == "conflict":
                    warnings.append("Text and image disagree - conservative approach taken")
            else:
                warnings.extend(img_warnings)
                warnings.append("Image analysis skipped due to validation errors")

        elif has_image and not is_dermatology:
            warnings.append(f"Image ignored - specialty is {result['specialty']}, not dermatology")

        elif not has_image and is_dermatology:
            warnings.append("Consider uploading an image for better dermatology assessment")

        return TriageResponse(
            specialty=result["specialty"],
            confidence=result["confidence"],
            differential_diagnosis=ddx,
            explanation=explanation,
            route=result.get("route"),
            modalities_used=modalities,
            image_analysis=image_analysis_resp,
            combined_assessment=combined_resp,
            warnings=warnings,
        )

    except Exception as e:
        logger.error("triage_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "healthy", "pipeline_loaded": _pipeline is not None}
