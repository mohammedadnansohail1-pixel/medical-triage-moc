"""
Emergency Agent - Safety-critical first check.

This agent runs EVERY turn before any other processing.
Uses rule-based detection (not LLM) for 100% reliability.
"""

from typing import Literal
from langchain_core.tools import tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.emergency_detector import detect_emergency
from .state import ConversationState, escalate_risk


@tool
def check_emergency(symptoms: list[str]) -> dict:
    """
    Check if symptoms indicate a medical emergency.
    
    Args:
        symptoms: List of symptom strings to check
        
    Returns:
        dict with is_emergency, matched_patterns, and message
    """
    # Check each symptom
    all_text = " ".join(symptoms).lower()
    result = detect_emergency(all_text)
    
    return {
        "is_emergency": result.get("is_emergency", False),
        "matched_patterns": result.get("matched_patterns", []),
        "message": result.get("message", ""),
    }


def run_emergency_check(state: ConversationState) -> dict:
    """
    Emergency check node for the graph.
    
    This runs on every turn to ensure we never miss an emergency.
    Returns state updates.
    """
    # Combine all symptoms + latest user message
    symptoms = state.get("symptoms_collected", [])
    
    # Also check the latest user message directly
    messages = state.get("messages", [])
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    
    all_text = symptoms + user_messages
    
    # Run check
    result = check_emergency.invoke({"symptoms": all_text})
    
    if result["is_emergency"]:
        emergency_response = {
            "role": "assistant",
            "content": (
                "⚠️ **MEDICAL EMERGENCY DETECTED**\n\n"
                f"{result['message']}\n\n"
                "**Please take immediate action:**\n"
                "• Call emergency services (911) immediately\n"
                "• Do not drive yourself to the hospital\n"
                "• Stay calm and follow dispatcher instructions\n\n"
                "This conversation will end here for your safety."
            ),
        }
        
        return {
            "messages": [emergency_response],
            "risk_level": "emergency",
            "triage_complete": True,
            "current_agent": "emergency",
            "triage_result": {
                "specialty": "emergency",
                "urgency": "emergency",
                "matched_patterns": result["matched_patterns"],
                "recommendation": "Seek immediate emergency care",
            },
        }
    
    # No emergency - continue to next agent
    return {
        "current_agent": state.get("current_agent", "supervisor"),
    }


# Emergency patterns that should trigger immediate response
# These supplement the rule-based detector with conversation context
EMERGENCY_KEYWORDS = [
    "can't breathe",
    "difficulty breathing", 
    "chest pain radiating",
    "crushing chest",
    "sudden severe headache",
    "worst headache of my life",
    "numbness on one side",
    "face drooping",
    "slurred speech",
    "unconscious",
    "seizure",
    "severe bleeding",
    "suicidal",
    "want to die",
    "overdose",
    "poisoning",
]


def quick_emergency_scan(text: str) -> bool:
    """Fast keyword scan for emergency terms."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in EMERGENCY_KEYWORDS)
