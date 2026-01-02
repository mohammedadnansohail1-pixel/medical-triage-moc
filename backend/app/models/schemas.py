

# ============================================================
# PHASE 4: Iterative Triage Schemas
# ============================================================

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field


class TriageRound(BaseModel):
    """Result of a single triage iteration."""
    round_number: int
    symptoms_collected: List[str]
    num_symptoms: int
    prediction: str
    confidence: float
    top_3_predictions: List[Tuple[str, float]]
    question_asked: Optional[str] = None
    symptom_code_asked: Optional[str] = None


class TriageSession(BaseModel):
    """Full iterative triage session state."""
    session_id: str
    patient_age: int
    patient_sex: str
    initial_symptoms: List[str]
    current_symptoms: List[str]
    rounds: List[TriageRound] = Field(default_factory=list)
    final_prediction: Optional[str] = None
    final_confidence: float = 0.0
    is_complete: bool = False
    termination_reason: Optional[str] = None


class QuestionCandidate(BaseModel):
    """Potential follow-up question with scoring."""
    symptom_code: str
    symptom_text: str
    information_gain: float
    specialty_relevance: float = 0.0
    combined_score: float = 0.0


class IterativeTriageRequest(BaseModel):
    """Request to start iterative triage session."""
    symptoms: List[str]
    age: int = 30
    sex: str = "M"
    max_rounds: int = 5
    confidence_threshold: float = 0.85


class IterativeTriageResponse(BaseModel):
    """Response from iterative triage."""
    session_id: str
    current_round: int
    prediction: str
    confidence: float
    top_3: List[Dict[str, float]]
    is_complete: bool
    next_question: Optional[str] = None
    next_symptom_code: Optional[str] = None


class EvaluationMetrics(BaseModel):
    """Metrics for evaluating iterative triage."""
    accuracy_per_round: Dict[int, float]
    average_interaction_length: float
    evidence_recall: float
    final_accuracy: float
    confidence_gain: float
