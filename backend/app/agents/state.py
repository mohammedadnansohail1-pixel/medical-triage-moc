"""Conversation state for multiagent symptom checker."""

from typing import Annotated, Literal, TypedDict
from dataclasses import dataclass, field
import operator


class PatientInfo(TypedDict, total=False):
    """Patient demographic information."""
    age: int | None
    sex: Literal["male", "female"] | None


class ConversationState(TypedDict):
    """
    State that persists through the conversation graph.
    
    This state is passed between agents and updated at each step.
    Using Annotated with operator.add for messages allows automatic
    message accumulation across graph nodes.
    """
    # Conversation history - messages accumulate
    messages: Annotated[list[dict], operator.add]
    
    # Collected symptoms from conversation
    symptoms_collected: list[str]
    
    # Patient demographics (can be collected during chat)
    patient_info: PatientInfo
    
    # Current active agent
    current_agent: str
    
    # Risk level: unknown -> routine -> elevated -> urgent -> emergency
    risk_level: Literal["unknown", "routine", "elevated", "urgent", "emergency"]
    
    # Detected or suggested specialty
    specialty_hint: str | None
    
    # Whether triage is complete
    triage_complete: bool
    
    # Turn counter (safety limit)
    turn_count: int
    
    # Final triage result (populated when complete)
    triage_result: dict | None
    
    # Image data if provided
    image_base64: str | None
    
    # Warnings/notes accumulated during conversation
    warnings: list[str]


def create_initial_state(
    message: str,
    patient_info: PatientInfo | None = None,
    image_base64: str | None = None,
    session_id: str | None = None,
) -> ConversationState:
    """Create initial conversation state from first user message."""
    return ConversationState(
        messages=[{"role": "user", "content": message}],
        symptoms_collected=[],
        patient_info=patient_info or {},
        current_agent="supervisor",
        risk_level="unknown",
        specialty_hint=None,
        triage_complete=False,
        turn_count=1,
        triage_result=None,
        image_base64=image_base64,
        warnings=[],
    )


# Maximum turns before forcing professional consultation recommendation
MAX_CONVERSATION_TURNS = 10

# Risk level ordering for comparisons
RISK_LEVELS = {
    "unknown": 0,
    "routine": 1,
    "elevated": 2,
    "urgent": 3,
    "emergency": 4,
}


def escalate_risk(current: str, new: str) -> str:
    """Return the higher risk level (conservative escalation)."""
    if RISK_LEVELS.get(new, 0) > RISK_LEVELS.get(current, 0):
        return new
    return current
