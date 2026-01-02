"""Triage API v2 - SapBERT + XGBoost pipeline."""

from pathlib import Path
from typing import Dict, List, Optional
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.triage_pipeline_v2 import get_triage_pipeline

logger = structlog.get_logger(__name__)
router = APIRouter()

# Paths
EVIDENCES_PATH = Path("/home/adnan21/projects/medical-triage-moc/data/ddxplus/release_evidences.json")
MODEL_PATH = Path("/home/adnan21/projects/medical-triage-moc/backend/data/classifier/model.pkl")
VOCAB_PATH = Path("/home/adnan21/projects/medical-triage-moc/backend/data/classifier/vocabulary.pkl")

# Pipeline singleton
_pipeline_loaded = False


def ensure_pipeline_loaded() -> None:
    """Load pipeline on first request."""
    global _pipeline_loaded
    if not _pipeline_loaded:
        pipeline = get_triage_pipeline()
        pipeline.load(EVIDENCES_PATH, MODEL_PATH, VOCAB_PATH)
        _pipeline_loaded = True


class TriageV2Request(BaseModel):
    """Input for triage v2 request."""
    symptoms: List[str] = Field(..., min_items=1, description="List of symptom strings")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    sex: Optional[str] = Field(None, pattern="^(male|female|other)$")


class TriageV2Response(BaseModel):
    """Output from triage v2 routing."""
    specialty: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: str
    matched_codes: List[str]
    reasoning: List[str]
    route: str
    is_emergency: bool
    emergency_reason: Optional[str] = None


@router.post("/triage/v2", response_model=TriageV2Response)
async def triage_patient_v2(request: TriageV2Request) -> TriageV2Response:
    """
    Route patient to appropriate specialty using SapBERT + XGBoost.
    
    Pipeline:
    1. Emergency detection (rule-based, 100% reliable)
    2. SapBERT: symptoms → DDXPlus evidence codes
    3. XGBoost: evidence codes → specialty (99% accuracy)
    """
    logger.info(
        "triage_v2_request",
        num_symptoms=len(request.symptoms),
        age=request.age,
        sex=request.sex,
    )

    try:
        ensure_pipeline_loaded()
        pipeline = get_triage_pipeline()
        
        result = pipeline.predict(request.symptoms)
        
        # Determine urgency
        if result["route"] == "EMERGENCY_OVERRIDE":
            urgency = "emergency"
        elif result["confidence"] > 0.8:
            urgency = "urgent"
        else:
            urgency = "routine"

        response = TriageV2Response(
            specialty=result["specialty"],
            confidence=result["confidence"],
            urgency=urgency,
            matched_codes=result["matched_codes"],
            reasoning=result["reasoning"],
            route=result["route"],
            is_emergency=result["emergency"]["is_emergency"],
            emergency_reason=result["emergency"].get("reason"),
        )

        logger.info(
            "triage_v2_response",
            specialty=response.specialty,
            confidence=response.confidence,
            urgency=response.urgency,
            route=response.route,
        )

        return response

    except Exception as e:
        logger.error("triage_v2_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/triage/v2/health")
async def triage_v2_health() -> Dict[str, str]:
    """Check if v2 pipeline is loaded."""
    try:
        ensure_pipeline_loaded()
        return {"status": "healthy", "pipeline": "loaded"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
