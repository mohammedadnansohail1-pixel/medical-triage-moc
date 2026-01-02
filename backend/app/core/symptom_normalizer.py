"""
Symptom Normalizer - Fuzzy matching + synonym expansion.
CPU only, no VRAM.
"""

from typing import List, Dict
from rapidfuzz import fuzz, process

MEDICAL_SYMPTOMS: Dict[str, List[str]] = {
    # Cardiac
    "chest pain": [
        "chest hurts", "chest is hurting", "pain in chest", "chest ache",
        "my chest hurts", "hurting chest", "sore chest", "chest discomfort",
        "chest is painful", "sharp chest pain"
    ],
    "palpitations": [
        "heart racing", "heart pounding", "heart beating fast", "rapid heartbeat",
        "fluttering heart", "skipped heartbeat", "heart skipping", "heart flutter"
    ],
    "chest tightness cardiac": [
        "tight chest", "chest tightness", "pressure in chest", "chest pressure",
        "squeezing chest", "heavy chest", "chest feels tight"
    ],
    
    # Respiratory  
    "dyspnea shortness of breath": [
        "hard to breathe", "difficulty breathing", "can't breathe", "short of breath",
        "breathing difficulty", "out of breath", "trouble breathing", "breathless",
        "cant breathe", "cannot breathe", "struggling to breathe"
    ],
    "cough": [
        "coughing", "bad cough", "dry cough", "persistent cough", "keep coughing"
    ],
    "productive cough sputum": [
        "coughing up mucus", "coughing up phlegm", "wet cough", "mucus cough"
    ],
    "wheezing bronchospasm": [
        "wheezing", "wheeze", "whistling breath", "noisy breathing"
    ],
    
    # GI
    "abdominal pain": [
        "stomach pain", "stomach hurts", "belly pain", "tummy ache", "gut pain",
        "stomach ache", "abdominal ache", "pain in stomach", "my stomach hurts",
        "stomache ache", "stomachache"
    ],
    "nausea": [
        "nauseous", "feel sick", "queasy", "feel like throwing up", "sick to stomach",
        "nauseus", "nauseated", "feeling nauseous", "feeling sick", "feeling nauseus"
    ],
    "vomiting": [
        "throwing up", "vomiting", "puking", "being sick", "threw up",
        "keep throwing up", "can't stop vomiting"
    ],
    "gastroesophageal reflux GERD": [
        "heartburn", "acid reflux", "reflux", "burning stomach", "acidic taste",
        "acid indigestion", "burning in chest after eating"
    ],
    "abdominal bloating distension": [
        "bloated", "bloating", "swollen belly", "distended", "gassy", "full feeling",
        "stomach bloated", "belly bloated"
    ],
    "diarrhea": [
        "diarrhea", "loose stool", "watery stool", "runny stool", "the runs"
    ],
    
    # Neurology
    "headache cephalgia": [
        "headache", "head hurts", "head pain", "bad headache", "head ache",
        "pounding head", "throbbing head", "headach", "my head hurts",
        "head is pounding", "my head is pounding"
    ],
    "migraine": [
        "migraine", "migraine headache", "severe headache with nausea"
    ],
    "dizziness vertigo": [
        "dizzy", "dizziness", "dizzy spells", "vertigo", "room spinning",
        "feeling dizzy", "head spinning", "lightheaded", "light headed"
    ],
    "syncope": [
        "fainted", "passed out", "blacked out", "fainting", "lost consciousness"
    ],
    "paresthesia numbness tingling": [
        "numbness", "tingling", "pins and needles", "numb", "prickling",
        "numbness in hands", "tingling fingers", "numb feet", "numb hands",
        "arm is numb", "leg is numb", "fingers are numb", "toes are numb",
        "hand is numb", "foot is numb"
    ],
    "gait instability ataxia": [
        "balance problems", "unsteady", "trouble walking", "off balance",
        "wobbly", "can't walk straight", "losing balance"
    ],
    "photophobia": [
        "sensitivity to light", "light sensitivity", "light hurts eyes",
        "bright light bothers", "eyes hurt in light"
    ],
    
    # Dermatology
    "skin rash dermatitis": [
        "rash", "skin rash", "rash on skin", "breaking out", "red rash",
        "rash on arm", "rash on leg", "rash on body", "got a rash"
    ],
    "pruritus itching": [
        "itchy", "itching", "itchy skin", "skin itching", "scratching",
        "itch", "can't stop scratching", "really itchy", "so itchy",
        "skin is itchy", "skin is really itchy", "very itchy"
    ],
    "urticaria hives": [
        "hives", "welts", "raised bumps", "allergic rash"
    ],
    "papules skin lesions": [
        "bumps", "red bumps", "pimples", "spots", "skin bumps", "lumps on skin"
    ],
    "skin inflammation dermatitis": [
        "skin irritation", "irritated skin", "red skin", "inflamed skin"
    ],
    
    # General/Constitutional
    "fatigue malaise": [
        "tired", "fatigue", "exhausted", "no energy", "feeling tired",
        "worn out", "drained", "sluggish", "low energy", "always tired"
    ],
    "fever pyrexia": [
        "fever", "high temperature", "feverish", "burning up", "chills and fever",
        "mild fever", "low grade fever", "have a fever"
    ],
    "myalgia": [
        "muscle pain", "body aches", "sore muscles", "aching muscles",
        "muscle ache", "achy", "muscles hurt", "body pain"
    ],
    "arthralgia": [
        "joint pain", "sore joints", "aching joints", "painful joints"
    ],
    "pharyngitis sore throat": [
        "sore throat", "throat pain", "throat hurts", "painful swallowing"
    ],
}

# Build reverse index
_PHRASE_TO_MEDICAL: Dict[str, str] = {}
_ALL_PHRASES: List[str] = []

for medical_term, patient_phrases in MEDICAL_SYMPTOMS.items():
    for phrase in patient_phrases:
        _PHRASE_TO_MEDICAL[phrase.lower()] = medical_term
        _ALL_PHRASES.append(phrase.lower())


def normalize_symptom(symptom: str, threshold: int = 60) -> str:
    """Normalize symptom using fuzzy matching. Default threshold=60."""
    symptom_lower = symptom.lower().strip()
    
    # Exact match first
    if symptom_lower in _PHRASE_TO_MEDICAL:
        return f"{symptom} {_PHRASE_TO_MEDICAL[symptom_lower]}"
    
    # Fuzzy match
    match = process.extractOne(
        symptom_lower,
        _ALL_PHRASES,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )
    
    if match:
        matched_phrase, score, _ = match
        medical_term = _PHRASE_TO_MEDICAL[matched_phrase]
        return f"{symptom} {medical_term}"
    
    return symptom


def normalize_symptoms(symptoms: List[str], threshold: int = 60) -> List[str]:
    """Normalize list of symptoms with fuzzy matching."""
    return [normalize_symptom(s, threshold) for s in symptoms]
