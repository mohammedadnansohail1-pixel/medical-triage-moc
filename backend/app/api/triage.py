"""
Triage API Endpoint.

POST /api/v1/triage - Process symptoms and return triage results with explanations.
"""

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


def get_pipeline() -> TriagePipelineV2:
    """Get or initialize pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = get_triage_pipeline()
        _pipeline.load(
            evidences_path=Path("/home/adnan21/projects/medical-triage-moc/data/ddxplus/release_evidences.json"),
            model_path=Path("data/classifier/model.pkl"),
            vocab_path=Path("data/classifier/vocabulary.pkl"),
            enable_explanations=True,
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
