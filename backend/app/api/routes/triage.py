"""Triage API endpoints."""

from typing import Dict, List, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.entity_extractor import extract_entities
from app.core.ensemble import ensemble_router

logger = structlog.get_logger(__name__)

router = APIRouter()


class TriageRequest(BaseModel):
    """Input for triage request."""

    symptoms: str = Field(..., min_length=1, description="Patient symptoms description")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    sex: Optional[str] = Field(None, pattern="^(male|female|other)$")
    medical_history: Optional[List[str]] = Field(default_factory=list)


class TriageResponse(BaseModel):
    """Output from triage routing."""

    primary_specialty: str
    secondary_specialty: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: str = Field(..., pattern="^(emergency|urgent|routine)$")
    extracted_symptoms: List[str]
    negated_symptoms: List[str]
    duration: Optional[str]
    severity: Optional[str]
    reasoning: List[str]
    recommendations: List[str]
    component_scores: Dict[str, Dict[str, float]]


@router.post("/triage", response_model=TriageResponse)
async def triage_patient(request: TriageRequest) -> TriageResponse:
    """
    Route patient to appropriate medical specialty.
    
    Uses ensemble of: Knowledge Graph + LLM + Rule-based routing.
    """
    logger.info(
        "triage_request_received",
        symptom_length=len(request.symptoms),
        age=request.age,
        sex=request.sex,
    )

    try:
        # 1. Extract entities from symptom text
        entities = extract_entities(request.symptoms)

        # 2. Route using ensemble
        result = await ensemble_router.route(
            entities=entities,
            age=request.age,
            sex=request.sex,
            medical_history=request.medical_history,
        )

        # 3. Build response
        response = TriageResponse(
            primary_specialty=result.primary_specialty,
            secondary_specialty=result.secondary_specialty,
            confidence=result.confidence,
            urgency=result.urgency,
            extracted_symptoms=result.extracted_symptoms,
            negated_symptoms=entities.negated_symptoms,
            duration=entities.duration,
            severity=entities.severity,
            reasoning=result.reasoning,
            recommendations=result.recommendations,
            component_scores=result.component_scores,
        )

        logger.info(
            "triage_response_sent",
            primary_specialty=response.primary_specialty,
            urgency=response.urgency,
            confidence=response.confidence,
        )

        return response

    except Exception as e:
        logger.error("triage_request_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Triage failed: {str(e)}")
