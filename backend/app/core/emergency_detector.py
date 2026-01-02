"""
Emergency Detection - Rule-based override for safety-critical symptoms.
100% reliable, no ML uncertainty. Always runs before specialty routing.
"""

import re
from typing import Dict, List, Tuple
import structlog

logger = structlog.get_logger(__name__)


# Emergency patterns - if ANY match, route to emergency
EMERGENCY_PATTERNS: List[Tuple[str, str]] = [
    # Cardiac emergencies
    (r"chest\s*(pain|pressure|tight|crush|squeeze).*breath", "cardiac_emergency"),
    (r"\bheart\s*attack\b", "cardiac_emergency"),
    (r"\bmyocardial\b", "cardiac_emergency"),
    (r"chest.*radiat.*(arm|jaw|back)", "cardiac_emergency"),
    
    # Stroke signs (FAST)
    (r"(face|arm|leg).*(numb|weak|droop).*sudden", "stroke"),
    (r"slur.*speech|speech.*slur|can'?t\s*speak", "stroke"),
    (r"sudden.*(confusion|trouble.*understand)", "stroke"),
    (r"sudden.*(vision|blind|see)", "stroke"),
    (r"sudden.*severe.*headache", "stroke"),
    (r"\bstroke\b", "stroke"),
    (r"\btia\b|transient.*ischemic", "stroke"),
    
    # Breathing emergencies
    (r"(can'?t|cannot|unable).*breath", "respiratory_emergency"),
    (r"(choking|choke|airway.*block)", "respiratory_emergency"),
    (r"(anaphyla|severe.*allerg).*breath", "anaphylaxis"),
    (r"lips.*(blue|purple)|cyan", "respiratory_emergency"),
    
    # Severe bleeding
    (r"(severe|heavy|won'?t\s*stop).*bleed", "hemorrhage"),
    (r"blood.*(everywhere|lot|pool)", "hemorrhage"),
    (r"cough.*blood|blood.*cough", "hemorrhage"),
    (r"vomit.*blood|blood.*vomit", "hemorrhage"),
    
    # Loss of consciousness
    (r"(unconscious|passed\s*out|faint.*not\s*wake)", "unconscious"),
    (r"\bseizure\b|\bconvuls|\bfitting\b", "seizure"),
    
    # Trauma
    (r"(car|vehicle|accident).*injur", "trauma"),
    (r"(fall|fell).*head.*(hit|hurt|injur)", "head_trauma"),
    (r"(gun|stab|knife).*wound", "trauma"),
    
    # Overdose/Poisoning
    (r"(overdose|od'?d|too\s*many\s*pills)", "overdose"),
    (r"(poison|toxic|swallow.*chemical)", "poisoning"),
    
    # Suicidal
    (r"(suicid|kill\s*myself|end\s*my\s*life|want\s*to\s*die)", "psychiatric_emergency"),
    (r"(self\s*harm|cut\s*myself|hurt\s*myself)", "psychiatric_emergency"),
]

# Compile patterns for performance
COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), reason) for p, reason in EMERGENCY_PATTERNS]


def detect_emergency(text: str) -> Dict:
    """
    Check if text contains emergency indicators.
    
    Args:
        text: Patient symptom description
        
    Returns:
        Dict with is_emergency, reason, matched_pattern
    """
    text_lower = text.lower()
    
    for pattern, reason in COMPILED_PATTERNS:
        match = pattern.search(text_lower)
        if match:
            logger.warning(
                "emergency_detected",
                reason=reason,
                matched=match.group(),
            )
            return {
                "is_emergency": True,
                "reason": reason,
                "matched_text": match.group(),
                "action": "ROUTE_TO_EMERGENCY",
            }
    
    return {
        "is_emergency": False,
        "reason": None,
        "matched_text": None,
        "action": "CONTINUE_ROUTING",
    }


def check_emergency_keywords(symptoms: List[str]) -> Dict:
    """
    Check list of symptoms for emergency indicators.
    
    Args:
        symptoms: List of symptom strings
        
    Returns:
        Emergency detection result
    """
    combined_text = " ".join(symptoms)
    return detect_emergency(combined_text)

# Additional emergency combinations
EMERGENCY_COMBOS = [
    # Cardiac
    ({"chest", "arm"}, "cardiac_emergency"),  # MI classic
    ({"chest", "sweating"}, "cardiac_emergency"),
    ({"chest pain", "shortness of breath"}, "cardiac_emergency"),
    # Stroke
    ({"face", "droop"}, "stroke"),
    ({"weakness", "numbness"}, "stroke"),
    ({"sudden", "weakness"}, "stroke"),
    # Anaphylaxis - MUST catch before dermatology rules
    ({"breathing", "hives"}, "anaphylaxis"),
    ({"breathing", "swelling"}, "anaphylaxis"),
    ({"breath", "hives"}, "anaphylaxis"),
    ({"breath", "swelling"}, "anaphylaxis"),
    ({"difficulty breathing", "hives"}, "anaphylaxis"),
    ({"difficulty breathing", "swelling"}, "anaphylaxis"),
    ({"shortness of breath", "hives"}, "anaphylaxis"),
    ({"shortness of breath", "swelling"}, "anaphylaxis"),
]

def check_emergency_combos(symptoms: List[str]) -> Dict:
    """Check for emergency symptom combinations."""
    text = " ".join(symptoms).lower()
    for combo, reason in EMERGENCY_COMBOS:
        if all(word in text for word in combo):
            return {
                "is_emergency": True,
                "reason": reason,
                "matched_text": str(combo),
                "action": "ROUTE_TO_EMERGENCY",
            }
    return {"is_emergency": False, "reason": None, "matched_text": None, "action": "CONTINUE_ROUTING"}

# Update main function
_original_check = check_emergency_keywords
def check_emergency_keywords(symptoms: List[str]) -> Dict:
    # Try regex patterns first
    result = _original_check.__wrapped__(symptoms) if hasattr(_original_check, '__wrapped__') else detect_emergency(" ".join(symptoms))
    if result["is_emergency"]:
        return result
    # Try combinations
    return check_emergency_combos(symptoms)
