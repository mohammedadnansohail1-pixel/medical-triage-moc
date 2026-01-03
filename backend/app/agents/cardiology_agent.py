"""
Cardiology Agent - Handles cardiac-related symptoms.

Specializes in:
- Asking relevant cardiac questions
- Risk factor assessment
- Urgency determination for cardiac symptoms
"""

from typing import Any
from langchain_core.tools import tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .state import ConversationState, escalate_risk


# Cardiac-related symptom keywords
CARDIAC_KEYWORDS = [
    "chest pain", "chest pressure", "chest tightness", "chest discomfort",
    "heart", "palpitations", "racing heart", "irregular heartbeat",
    "shortness of breath", "breathless", "can't breathe", "out of breath",
    "dizzy", "lightheaded", "fainting", "passed out",
    "arm pain", "jaw pain", "sweating", "cold sweat",
    "fatigue", "tired", "exhausted", "weak",
    # Heart failure symptoms
    "swollen ankles", "ankle swelling", "leg swelling", "swollen legs",
    "can't lie flat", "need pillows to sleep", "wake up breathless",
    "fluid retention", "edema", "weight gain",
]


def is_cardiac_related(text: str) -> bool:
    """Check if text contains cardiac-related symptoms."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CARDIAC_KEYWORDS)


# Cardiac risk factors to assess
CARDIAC_RISK_FACTORS = [
    "high blood pressure",
    "diabetes",
    "high cholesterol", 
    "smoking",
    "family history of heart disease",
    "obesity",
    "sedentary lifestyle",
    "stress",
]


# Follow-up questions for cardiac symptoms
CARDIAC_QUESTIONS = [
    ("When did your symptoms start?", "onset"),
    ("Is the chest pain sharp, dull, or pressure-like?", "pain_quality"),
    ("Does the pain spread to your arm, jaw, neck, or back?", "radiation"),
    ("Are you experiencing shortness of breath?", "dyspnea"),
    ("Do you have any of these risk factors: high blood pressure, diabetes, high cholesterol, or smoking?", "risk_factors"),
    ("Does the pain get worse with physical activity or stress?", "exertion"),
    ("Have you experienced these symptoms before?", "history"),
]


# Red flags that should escalate urgency
CARDIAC_RED_FLAGS = [
    "radiating to arm",
    "radiating to jaw",
    "crushing pain",
    "pressure in chest",
    "cold sweat",
    "nausea with chest pain",
    "sudden onset",
    "worse with exertion",
    "shortness of breath at rest",
]


def check_cardiac_red_flags(symptoms: list[str], messages: list[dict]) -> list[str]:
    """Check for cardiac red flags in symptoms and conversation."""
    all_text = " ".join(symptoms).lower()
    for m in messages:
        if m.get("role") == "user":
            all_text += " " + m.get("content", "").lower()
    
    found_flags = []
    for flag in CARDIAC_RED_FLAGS:
        if flag in all_text:
            found_flags.append(flag)
    
    # Check for radiation patterns
    if "arm" in all_text and ("pain" in all_text or "chest" in all_text):
        if "radiating to arm" not in found_flags:
            found_flags.append("pain radiating to arm")
    
    if "jaw" in all_text and ("pain" in all_text or "chest" in all_text):
        if "radiating to jaw" not in found_flags:
            found_flags.append("pain radiating to jaw")
    
    return found_flags


@tool  
def assess_cardiac_risk(
    symptoms: list[str],
    age: int = 30,
    sex: str = "male",
    risk_factors: list[str] | None = None,
) -> dict:
    """
    Assess cardiac risk based on symptoms and risk factors.
    
    Args:
        symptoms: List of cardiac symptoms
        age: Patient age
        sex: Patient sex
        risk_factors: Known cardiac risk factors
        
    Returns:
        Risk assessment with urgency and recommendations
    """
    risk_factors = risk_factors or []
    
    # Calculate risk score
    risk_score = 0
    
    # Age factor
    if sex == "male" and age >= 45:
        risk_score += 1
    elif sex == "female" and age >= 55:
        risk_score += 1
    if age >= 65:
        risk_score += 1
    
    # Symptom severity
    symptoms_text = " ".join(symptoms).lower()
    
    high_risk_symptoms = ["chest pain", "pressure", "crushing", "radiating"]
    for symptom in high_risk_symptoms:
        if symptom in symptoms_text:
            risk_score += 2
    
    moderate_symptoms = ["palpitations", "shortness of breath", "dizzy", "fatigue"]
    for symptom in moderate_symptoms:
        if symptom in symptoms_text:
            risk_score += 1
    
    # Risk factors
    risk_score += len(risk_factors)
    
    # Determine urgency
    if risk_score >= 6:
        urgency = "urgent"
        recommendation = "Seek medical attention within 24 hours. Consider ER if symptoms worsen."
    elif risk_score >= 4:
        urgency = "semi-urgent"
        recommendation = "Schedule an appointment with a cardiologist within the next few days."
    elif risk_score >= 2:
        urgency = "elevated"
        recommendation = "See your primary care doctor soon for evaluation."
    else:
        urgency = "routine"
        recommendation = "Monitor symptoms. Schedule a routine check-up if they persist."
    
    return {
        "risk_score": risk_score,
        "urgency": urgency,
        "recommendation": recommendation,
        "risk_factors_identified": risk_factors,
        "age_risk": age >= 45 if sex == "male" else age >= 55,
    }


def format_cardiac_response(
    assessment: dict,
    symptoms: list[str],
    red_flags: list[str],
) -> str:
    """Format cardiac assessment as patient-friendly response."""
    urgency = assessment.get("urgency", "routine")
    recommendation = assessment.get("recommendation", "")
    risk_score = assessment.get("risk_score", 0)
    
    # Urgency icons
    urgency_icons = {
        "urgent": "🚨",
        "semi-urgent": "⚠️",
        "elevated": "📋",
        "routine": "✅",
    }
    icon = urgency_icons.get(urgency, "📋")
    
    response = f"""{icon} **Cardiac Assessment**

**Symptoms Reported:** {', '.join(symptoms)}
**Urgency Level:** {urgency.title()}

{recommendation}"""

    # Add red flag warning
    if red_flags:
        response += f"\n\n⚠️ **Concerning Signs Noted:**\n"
        for flag in red_flags:
            response += f"• {flag.title()}\n"
    
    # Add general cardiac advice
    response += """

**While Waiting for Medical Evaluation:**
- Rest and avoid strenuous activity
- Take aspirin if advised by a healthcare provider
- Keep track of when symptoms occur
- Note what makes symptoms better or worse

⚕️ **Important:** Cardiac symptoms can be serious. If you experience sudden severe chest pain, difficulty breathing, or feel like something is very wrong, call 911 immediately."""

    return response


def run_cardiology_node(state: ConversationState) -> dict:
    """
    Cardiology agent node for the graph.
    
    Handles cardiac-related questions and risk assessment.
    """
    symptoms = state.get("symptoms_collected", [])
    messages = state.get("messages", [])
    patient_info = state.get("patient_info", {})
    current_risk = state.get("risk_level", "unknown")
    
    # Check for red flags first
    red_flags = check_cardiac_red_flags(symptoms, messages)
    
    # If multiple red flags, escalate immediately
    if len(red_flags) >= 2:
        response = """🚨 **Important: Multiple Cardiac Warning Signs Detected**

Based on your symptoms, I'm concerned about potential cardiac issues.

**Please take this seriously:**
- If you're having chest pain RIGHT NOW with arm/jaw pain or shortness of breath, **call 911 immediately**
- Do not drive yourself to the hospital
- Chew an aspirin (325mg) if you're not allergic

If your symptoms are not acute right now, please see a cardiologist or visit an urgent care within 24 hours.

Would you like me to continue with a detailed assessment, or are you experiencing acute symptoms now?"""
        
        return {
            "messages": [{"role": "assistant", "content": response}],
            "risk_level": "urgent",
            "specialty_hint": "cardiology",
            "current_agent": "cardiology",
        }
    
    # Determine which questions have been asked
    asked_topics = set()
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "").lower()
            for question, topic in CARDIAC_QUESTIONS:
                if any(word in content for word in question.lower().split()[:3]):
                    asked_topics.add(topic)
    
    # Find next question to ask
    remaining = [(q, t) for q, t in CARDIAC_QUESTIONS if t not in asked_topics]
    
    if remaining and len(asked_topics) < 4:
        # Still gathering information
        question, _ = remaining[0]
        response = f"To better assess your cardiac symptoms:\n\n{question}"
        
        return {
            "messages": [{"role": "assistant", "content": response}],
            "current_agent": "cardiology",
            "specialty_hint": "cardiology",
        }
    
    # Enough info gathered - run assessment
    assessment = assess_cardiac_risk.invoke({
        "symptoms": symptoms,
        "age": patient_info.get("age", 30),
        "sex": patient_info.get("sex", "male"),
        "risk_factors": [],  # Could extract from conversation
    })
    
    response = format_cardiac_response(assessment, symptoms, red_flags)
    
    # Map urgency to risk
    urgency_to_risk = {
        "urgent": "urgent",
        "semi-urgent": "elevated",
        "elevated": "elevated",
        "routine": "routine",
    }
    new_risk = urgency_to_risk.get(assessment.get("urgency", "routine"), "routine")
    
    return {
        "messages": [{"role": "assistant", "content": response}],
        "risk_level": escalate_risk(current_risk, new_risk),
        "specialty_hint": "cardiology",
        "current_agent": "cardiology",
        "triage_complete": True,
        "triage_result": assessment,
    }
