"""
Multiagent Symptom Checker

A conversational medical triage system using LangGraph.

Architecture:
- Supervisor Agent: Routes conversation, asks clarifying questions
- Emergency Agent: Detects emergency symptoms (rule-based, 100% reliable)
- Triage Agent: Runs full DDX pipeline when ready
- Dermatology Agent: Handles skin-related questions + image analysis
- Cardiology Agent: Handles cardiac symptom deep-dive
"""

from .state import (
    ConversationState,
    PatientInfo,
    create_initial_state,
    escalate_risk,
    MAX_CONVERSATION_TURNS,
    RISK_LEVELS,
)

__all__ = [
    "ConversationState",
    "PatientInfo", 
    "create_initial_state",
    "escalate_risk",
    "MAX_CONVERSATION_TURNS",
    "RISK_LEVELS",
]
