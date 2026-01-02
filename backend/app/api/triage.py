"""
Triage API Endpoint.

POST /api/v1/triage - Process symptoms and return triage results with explanations.
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.triage_pipeline_v2 import TriagePipelineV2, get_triage_pipeline

router = APIRouter(prefix="/api/v1", tags=["triage"])


class TriageRequest(BaseModel):
    """Request model for triage endpoint."""
    symptoms: List[str] = Field(..., min_length=1, description="List of symptoms")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    sex: Optional[str] = Field(None, pattern="^(male|female)$", description="Patient sex")
    include_explanation: bool = Field(True, description="Include LLM explanation")


class DifferentialDiagnosis(BaseModel):
    """Single differential diagnosis entry."""
    condition: str
    probability: float
    rank: int


class Explanation(BaseModel):
    """LLM-generated explanation."""
    text: str
    urgency: str  # emergency, urgent, routine
    next_steps: List[str]


class TriageResponse(BaseModel):
    """Response model for triage endpoint."""
    specialty: str
    confidence: float
    differential_diagnosis: List[DifferentialDiagnosis]
    explanation: Optional[Explanation] = None
    route: Optional[str] = None


# Pipeline singleton
_pipeline: Optional[TriagePipelineV2] = None


def get_data_paths():
    """Get data paths - works in both local and Docker."""
    # Check for Docker paths first
    if Path("/app/data/ddxplus/release_evidences.json").exists():
        base = Path("/app")
    else:
        # Local development
        base = Path(__file__).parent.parent.parent.parent
    
    return {
        "evidences": base / "data" / "ddxplus" / "release_evidences.json",
        "model": Path(__file__).parent.parent.parent / "data" / "classifier" / "model.pkl",
        "vocab": Path(__file__).parent.parent.parent / "data" / "classifier" / "vocabulary.pkl",
    }


def get_pipeline() -> TriagePipelineV2:
    """Get or initialize pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        paths = get_data_paths()
        
        _pipeline = get_triage_pipeline()
        
        # Check if LLM is enabled
        enable_llm = os.environ.get("ENABLE_LLM", "true").lower() == "true"
        
        _pipeline.load(
            evidences_path=paths["evidences"],
            model_path=paths["model"],
            vocab_path=paths["vocab"],
            enable_explanations=enable_llm,
        )
    return _pipeline


@router.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest) -> TriageResponse:
    """
    Process symptoms and return triage results.

    Returns specialty routing, differential diagnosis, and optional
    LLM-generated explanation with urgency level.
    """
    try:
        pipeline = get_pipeline()

        result = pipeline.predict(
            symptoms=request.symptoms,
            age=request.age,
            sex=request.sex,
            include_explanation=request.include_explanation,
        )

        # Build response
        ddx = [
            DifferentialDiagnosis(
                condition=d["condition"],
                probability=d["probability"],
                rank=i + 1,
            )
            for i, d in enumerate(result.get("differential_diagnosis", []))
        ]

        explanation = None
        if result.get("explanation"):
            exp = result["explanation"]
            explanation = Explanation(
                text=exp["text"],
                urgency=exp["urgency"],
                next_steps=exp["next_steps"],
            )

        return TriageResponse(
            specialty=result["specialty"],
            confidence=result["confidence"],
            differential_diagnosis=ddx,
            explanation=explanation,
            route=result.get("route"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "pipeline_loaded": _pipeline is not None}
