"""
Dermatology Agent - Handles skin-related symptoms and image analysis.

Specializes in:
- Asking relevant skin-related questions
- Analyzing skin images when provided
- Combining text + image for multimodal assessment
"""

from typing import Any
from langchain_core.tools import tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .state import ConversationState, escalate_risk


# Lazy-loaded components
_skin_classifier = None
_image_validator = None


def get_skin_classifier():
    """Lazy-load skin classifier."""
    global _skin_classifier
    if _skin_classifier is None:
        from core.skin_classifier import get_skin_classifier as _get_clf
        _skin_classifier = _get_clf()
    return _skin_classifier


def get_image_validator():
    """Lazy-load image validator."""
    global _image_validator
    if _image_validator is None:
        from core.image_validator import ImageValidator
        _image_validator = ImageValidator()
    return _image_validator


# Skin-related symptom keywords
SKIN_KEYWORDS = [
    "rash", "mole", "spot", "bump", "lesion", "itchy", "itch",
    "skin", "red", "swelling", "bruise", "blister", "hives",
    "acne", "pimple", "wart", "growth", "discoloration",
    "scaly", "flaky", "dry skin", "peeling", "sore",
]


def is_skin_related(text: str) -> bool:
    """Check if text contains skin-related symptoms."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SKIN_KEYWORDS)


# Follow-up questions for skin conditions
SKIN_QUESTIONS = [
    "How long have you had this skin condition?",
    "Is it itchy, painful, or neither?",
    "Has it changed in size, shape, or color recently?",
    "Have you noticed any similar spots elsewhere on your body?",
    "Do you have a photo of the affected area you can share?",
]


@tool
def analyze_skin_image(image_base64: str) -> dict:
    """
    Analyze a skin lesion image.
    
    Args:
        image_base64: Base64-encoded image data
        
    Returns:
        Analysis result with prediction, confidence, tier, and recommendations
    """
    # Validate image first
    validator = get_image_validator()
    validation = validator.validate(image_base64)
    
    if not validation.is_valid:
        return {
            "success": False,
            "error": "; ".join(validation.errors),
            "warnings": validation.warnings,
        }
    
    # Run classifier
    classifier = get_skin_classifier()
    result = classifier.predict(validation.image)
    
    return {
        "success": True,
        "prediction": result.get("prediction", "Unknown"),
        "prediction_label": result.get("prediction_label", "Unknown"),
        "confidence": result.get("confidence", 0.0),
        "tier": result.get("tier", "consider_evaluation"),
        "tier_display": result.get("tier_display", "Consider Evaluation"),
        "cancer_probability": result.get("cancer_probability", 0.0),
        "message": result.get("message", ""),
        "action": result.get("action", ""),
        "warnings": validation.warnings,
    }


def format_skin_analysis_response(result: dict) -> str:
    """Format skin analysis result as patient-friendly response."""
    if not result.get("success", False):
        return f"I couldn't analyze the image: {result.get('error', 'Unknown error')}. Could you try uploading a clearer photo?"
    
    prediction_label = result.get("prediction_label", "Unknown")
    confidence = result.get("confidence", 0) * 100
    tier = result.get("tier", "consider_evaluation")
    tier_display = result.get("tier_display", "Consider Evaluation")
    cancer_prob = result.get("cancer_probability", 0) * 100
    message = result.get("message", "")
    action = result.get("action", "")
    
    # Tier-specific formatting
    tier_icons = {
        "routine_monitoring": "✅",
        "consider_evaluation": "📋",
        "routine_referral": "📅",
        "urgent_referral": "⚠️",
    }
    icon = tier_icons.get(tier, "📋")
    
    response = f"""{icon} **Image Analysis Results**

**Most Likely:** {prediction_label} ({confidence:.0f}% confidence)
**Risk Level:** {tier_display}

{message}

**Recommended Action:** {action}"""

    # Add cancer warning if elevated
    if cancer_prob > 10:
        response += f"\n\n⚠️ Note: This image shows features that warrant professional evaluation (cancer probability: {cancer_prob:.0f}%)."
    
    # Add warnings if any
    warnings = result.get("warnings", [])
    if warnings:
        response += f"\n\n📝 Image quality notes: {'; '.join(warnings)}"
    
    response += "\n\n⚕️ **Important:** This is an AI screening tool, not a diagnosis. Please consult a dermatologist for proper evaluation."
    
    return response


def run_dermatology_node(state: ConversationState) -> dict:
    """
    Dermatology agent node for the graph.
    
    Handles skin-related questions and image analysis.
    """
    image_base64 = state.get("image_base64")
    symptoms = state.get("symptoms_collected", [])
    messages = state.get("messages", [])
    current_risk = state.get("risk_level", "unknown")
    
    # If we have an image, analyze it
    if image_base64:
        result = analyze_skin_image.invoke({"image_base64": image_base64})
        response_text = format_skin_analysis_response(result)
        
        # Map tier to risk level
        tier_to_risk = {
            "routine_monitoring": "routine",
            "consider_evaluation": "routine",
            "routine_referral": "elevated",
            "urgent_referral": "urgent",
        }
        new_risk = tier_to_risk.get(result.get("tier", "consider_evaluation"), "routine")
        
        # Check if we should complete triage or continue
        triage_complete = result.get("tier") in ["urgent_referral", "routine_referral"]
        
        return {
            "messages": [{
                "role": "assistant",
                "content": response_text,
            }],
            "risk_level": escalate_risk(current_risk, new_risk),
            "specialty_hint": "dermatology",
            "current_agent": "dermatology",
            "triage_complete": triage_complete,
            "triage_result": result if triage_complete else None,
        }
    
    # No image - ask follow-up questions
    # Determine which question to ask based on conversation history
    asked_questions = []
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            for q in SKIN_QUESTIONS:
                if q.lower() in content.lower():
                    asked_questions.append(q)
    
    # Find next question to ask
    remaining_questions = [q for q in SKIN_QUESTIONS if q not in asked_questions]
    
    if remaining_questions:
        # Ask next question
        response_text = f"To help assess your skin condition:\n\n{remaining_questions[0]}"
        
        # Prompt for image if not asked yet
        if "photo" not in " ".join(asked_questions).lower() and len(asked_questions) >= 2:
            response_text += "\n\nIf you can share a clear photo of the affected area, I can provide a more detailed analysis."
    else:
        # All questions asked, provide general advice
        response_text = """Based on your description, I recommend:

1. **Monitor the area** - Note any changes in size, color, or texture
2. **Avoid irritants** - Don't scratch or apply unknown products
3. **See a dermatologist** - For persistent or concerning symptoms

Would you like me to provide a full triage assessment based on all your symptoms?"""
    
    return {
        "messages": [{
            "role": "assistant",
            "content": response_text,
        }],
        "current_agent": "dermatology",
        "specialty_hint": "dermatology",
    }
