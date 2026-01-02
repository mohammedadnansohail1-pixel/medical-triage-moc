"""
Triage Agent - Runs the full DDX pipeline.

Called when supervisor has collected enough symptoms to make a triage decision.
Wraps our existing high-accuracy pipeline.
"""

from typing import Any
from langchain_core.tools import tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .state import ConversationState, escalate_risk


# Lazy-loaded pipeline
_triage_pipeline = None


def get_triage_pipeline():
    """Lazy-load the triage pipeline."""
    global _triage_pipeline
    if _triage_pipeline is None:
        from core.triage_pipeline_v2 import TriagePipelineV2
        _triage_pipeline = TriagePipelineV2()
    return _triage_pipeline


@tool
def run_triage_pipeline(
    symptoms: list[str],
    age: int = 30,
    sex: str = "male",
) -> dict:
    """
    Run the full medical triage pipeline.
    
    Args:
        symptoms: List of patient symptoms
        age: Patient age (default 30)
        sex: Patient sex (male/female)
        
    Returns:
        Triage result with specialty, urgency, and differential diagnosis
    """
    pipeline = get_triage_pipeline()
    
    result = pipeline.triage(
        symptoms=symptoms,
        age=age,
        sex=sex,
    )
    
    return {
        "specialty": result.get("specialty", "general"),
        "confidence": result.get("confidence", 0.0),
        "urgency": result.get("urgency", "routine"),
        "differential_diagnosis": result.get("differential_diagnosis", []),
        "route": result.get("route", "ML_CLASSIFICATION"),
        "matched_codes": result.get("matched_codes", []),
    }


def format_triage_response(result: dict, symptoms: list[str]) -> str:
    """Format triage result as patient-friendly response."""
    specialty = result.get("specialty", "general medicine")
    urgency = result.get("urgency", "routine")
    ddx = result.get("differential_diagnosis", [])
    
    # Urgency-specific advice
    urgency_advice = {
        "emergency": "⚠️ Please seek immediate emergency care.",
        "urgent": "📋 We recommend seeing a doctor within 24-48 hours.",
        "semi-urgent": "📅 Please schedule an appointment within the next few days.",
        "routine": "✅ You can schedule a routine appointment at your convenience.",
    }
    
    advice = urgency_advice.get(urgency, urgency_advice["routine"])
    
    # Format differential diagnosis
    conditions_text = ""
    if ddx:
        top_conditions = ddx[:3]  # Top 3
        conditions_list = [
            f"• {c.get('condition', 'Unknown')} ({c.get('probability', 0)*100:.0f}% likelihood)"
            for c in top_conditions
        ]
        conditions_text = "\n".join(conditions_list)
    
    response = f"""Based on your symptoms ({', '.join(symptoms)}), here's my assessment:

**Recommended Specialty:** {specialty.replace('_', ' ').title()}

**Urgency Level:** {urgency.title()}
{advice}

**Possible Conditions to Discuss with Your Doctor:**
{conditions_text if conditions_text else "• Further evaluation needed"}

⚕️ **Important:** This is not a diagnosis. Please consult with a healthcare professional for proper evaluation and treatment."""

    return response


def run_triage_node(state: ConversationState) -> dict:
    """
    Triage node for the graph.
    
    Runs when supervisor determines we have enough info for triage.
    """
    symptoms = state.get("symptoms_collected", [])
    patient_info = state.get("patient_info", {})
    
    if not symptoms:
        # Not enough info yet
        return {
            "messages": [{
                "role": "assistant", 
                "content": "I need to know more about your symptoms before I can help. What symptoms are you experiencing?"
            }],
            "current_agent": "supervisor",
        }
    
    # Run pipeline
    result = run_triage_pipeline.invoke({
        "symptoms": symptoms,
        "age": patient_info.get("age", 30),
        "sex": patient_info.get("sex", "male"),
    })
    
    # Map urgency to risk level
    urgency_to_risk = {
        "emergency": "emergency",
        "urgent": "urgent", 
        "semi-urgent": "elevated",
        "routine": "routine",
    }
    new_risk = urgency_to_risk.get(result.get("urgency", "routine"), "routine")
    current_risk = state.get("risk_level", "unknown")
    
    # Format response
    response_text = format_triage_response(result, symptoms)
    
    return {
        "messages": [{
            "role": "assistant",
            "content": response_text,
        }],
        "risk_level": escalate_risk(current_risk, new_risk),
        "specialty_hint": result.get("specialty"),
        "triage_complete": True,
        "current_agent": "triage",
        "triage_result": result,
    }
