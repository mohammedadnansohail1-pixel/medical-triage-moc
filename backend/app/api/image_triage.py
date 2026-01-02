"""
Image Triage API Endpoint with 4-tier risk stratification.

Aligned with NICE NG12 suspected cancer pathways.
"""

import base64
from io import BytesIO
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from app.core.skin_classifier import get_skin_classifier, CLASS_INFO, TIER_INFO, RiskTier

router = APIRouter(prefix="/api/v1", tags=["image-triage"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024


class PredictionItem(BaseModel):
    """Single prediction entry."""
    class_code: str
    class_name: str
    probability: float
    risk_level: str
    is_cancer: bool
    is_precancer: bool


class ProbabilitySummary(BaseModel):
    """Aggregate probability summary."""
    benign: float
    precancer: float
    cancer_total: float
    melanoma: float


class RiskAssessment(BaseModel):
    """4-tier risk assessment aligned with NICE guidelines."""
    tier: str
    tier_display: str
    color: str
    timeframe: str
    message: str
    action: str
    reasons: List[str]


class ImageAnalysis(BaseModel):
    """Complete image analysis results."""
    prediction: str
    prediction_label: str
    confidence: float
    description: str
    icd10: str
    all_predictions: List[PredictionItem]
    probability_summary: ProbabilitySummary
    risk_assessment: RiskAssessment
    disclaimer: str


class ImageTriageResponse(BaseModel):
    """Response model for image triage endpoint."""
    specialty: str = "dermatology"
    route: str = "IMAGE_CLASSIFICATION"
    image_analysis: ImageAnalysis
    symptoms_provided: List[str]
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


class Base64ImageRequest(BaseModel):
    """Request model for base64 image upload."""
    image_base64: str = Field(..., description="Base64 encoded image")
    symptoms: Optional[str] = Field(None, description="Comma-separated symptoms")
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, pattern="^(male|female)$")


def validate_image(image: Image.Image) -> None:
    """Validate image dimensions."""
    if image.width < 50 or image.height < 50:
        raise HTTPException(status_code=400, detail="Image too small (min 50x50)")
    if image.width > 4096 or image.height > 4096:
        raise HTTPException(status_code=400, detail="Image too large (max 4096x4096)")


def build_response(
    analysis: dict, 
    symptoms: List[str],
    age: Optional[int] = None,
    sex: Optional[str] = None
) -> ImageTriageResponse:
    """Build API response from analysis."""
    return ImageTriageResponse(
        specialty="dermatology",
        route="IMAGE_CLASSIFICATION",
        image_analysis=ImageAnalysis(
            prediction=analysis["prediction"],
            prediction_label=analysis["prediction_label"],
            confidence=analysis["confidence"],
            description=analysis["description"],
            icd10=analysis["icd10"],
            all_predictions=[PredictionItem(**p) for p in analysis["all_predictions"]],
            probability_summary=ProbabilitySummary(**analysis["probability_summary"]),
            risk_assessment=RiskAssessment(**analysis["risk_assessment"]),
            disclaimer=analysis["disclaimer"],
        ),
        symptoms_provided=symptoms,
        patient_age=age,
        patient_sex=sex,
    )


@router.post("/triage/image", response_model=ImageTriageResponse)
async def triage_image(
    image: UploadFile = File(..., description="Skin lesion image (JPEG/PNG)"),
    symptoms: Optional[str] = Form(None, description="Comma-separated symptoms"),
    age: Optional[int] = Form(None, ge=0, le=120),
    sex: Optional[str] = Form(None),
):
    """
    Analyze skin lesion image with 4-tier risk stratification.
    
    Risk tiers (aligned with NICE NG12):
    - **routine_monitoring**: Clearly benign, self-monitor
    - **consider_evaluation**: Low concern, GP review if worried
    - **routine_referral**: Schedule dermatology appointment
    - **urgent_referral**: 2-week suspected cancer pathway
    """
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid type. Allowed: {ALLOWED_TYPES}")

    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    try:
        pil_image = Image.open(BytesIO(contents))
        validate_image(pil_image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()] if symptoms else []

    try:
        classifier = get_skin_classifier()
        analysis = classifier.classify_with_context(
            image=pil_image,
            symptoms=symptom_list,
            age=age,
            sex=sex,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    return build_response(analysis, symptom_list, age, sex)


@router.post("/triage/image/base64", response_model=ImageTriageResponse)
async def triage_image_base64(request: Base64ImageRequest):
    """Analyze skin lesion image (base64) with 4-tier risk stratification."""
    try:
        image_data = request.image_base64
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    try:
        pil_image = Image.open(BytesIO(image_bytes))
        validate_image(pil_image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    symptom_list = [s.strip() for s in request.symptoms.split(",") if s.strip()] if request.symptoms else []

    try:
        classifier = get_skin_classifier()
        analysis = classifier.classify_with_context(
            image=pil_image,
            symptoms=symptom_list,
            age=request.age,
            sex=request.sex,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

    return build_response(analysis, symptom_list, request.age, request.sex)


@router.get("/skin-classes")
async def get_skin_classes():
    """Get information about skin lesion classes and risk tiers."""
    return {
        "classes": CLASS_INFO,
        "risk_tiers": {tier.value: TIER_INFO[tier] for tier in RiskTier},
        "urgent_cancer_classes": ["MEL", "SCC"],
        "routine_cancer_classes": ["BCC"],
        "precancer_classes": ["AK"],
        "benign_classes": ["BKL", "DF", "NV", "VASC"],
    }
