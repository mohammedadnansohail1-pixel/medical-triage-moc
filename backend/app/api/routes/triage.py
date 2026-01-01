"""Triage API endpoints."""

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class TriageRequest(BaseModel):
    """Input for triage request."""

    symptoms: str = Field(..., min_length=1, description="Patient symptoms")
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
    reasoning: List[str]
    recommendations: List[str]


@router.post("/triage", response_model=TriageResponse)
async def triage_patient(request: TriageRequest) -> TriageResponse:
    """
    Route patient to appropriate medical specialty.
    
    Uses ensemble of: Knowledge Graph + LLM + Rule-based routing.
    """
    # TODO: Implement full triage logic
    # For now, return placeholder response
    return TriageResponse(
        primary_specialty="general_medicine",
        secondary_specialty=None,
        confidence=0.5,
        urgency="routine",
        extracted_symptoms=["placeholder"],
        reasoning=["Triage logic not yet implemented"],
        recommendations=["Please implement triage pipeline"],
    )
