"""
LLM-based Explanation Generator for Triage Results.

Generates patient-friendly natural language explanations
for specialty routing and differential diagnosis.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a medical triage assistant. Generate clear, patient-friendly explanations.

Rules:
- Use simple language a patient can understand
- Be reassuring but accurate  
- Never diagnose - only explain what the AI system suggests
- Keep explanations to 2-3 sentences
- Always recommend consulting a healthcare professional

Respond with valid JSON:
{"explanation": "2-3 sentence explanation", "urgency": "emergency|urgent|routine", "next_steps": ["step1", "step2"]}"""


@dataclass
class ExplanationResult:
    """Result from explanation generation."""
    explanation: str
    urgency: str  # emergency, urgent, routine
    next_steps: List[str]
    raw_response: Optional[str] = None


class ExplanationGenerator:
    """Generates natural language explanations using local LLM (Ollama)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def _build_prompt(
        self,
        symptoms: List[str],
        specialty: str,
        confidence: float,
        differential_diagnosis: List[Dict],
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> str:
        """Build structured prompt for LLM."""
        
        top_condition = "unknown"
        if differential_diagnosis:
            top = differential_diagnosis[0]
            top_condition = f"{top['condition']} ({top['probability']:.0%})"
        
        patient_info = ""
        if age:
            patient_info = f"{age}yo "
        if sex:
            patient_info += f"{sex} "
        
        return f"""{SYSTEM_PROMPT}

{patient_info}patient with: {', '.join(symptoms)}

Triage result:
- Specialty: {specialty.replace('_', ' ')}
- Top condition: {top_condition}
- Confidence: {confidence:.0%}

Generate JSON:"""
    
    def generate(
        self,
        symptoms: List[str],
        specialty: str,
        confidence: float,
        differential_diagnosis: List[Dict],
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> ExplanationResult:
        """Generate explanation for triage result."""
        prompt = self._build_prompt(
            symptoms=symptoms,
            specialty=specialty,
            confidence=confidence,
            differential_diagnosis=differential_diagnosis,
            age=age,
            sex=sex,
        )
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",  # Force valid JSON output
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 300,
                    }
                },
            )
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get("response", "")
            
            logger.info(
                "explanation_generated",
                model=self.model,
                tokens=result.get("eval_count"),
                duration_ms=round(result.get("eval_duration", 0) / 1e6),
            )
            
            return self._parse_response(raw_response, specialty, differential_diagnosis)
            
        except httpx.HTTPError as e:
            logger.error("ollama_request_failed", error=str(e))
            return self._fallback_explanation(specialty, differential_diagnosis)
        except Exception as e:
            logger.error("explanation_generation_failed", error=str(e))
            return self._fallback_explanation(specialty, differential_diagnosis)
    
    def _parse_response(
        self, 
        raw_response: str, 
        specialty: str,
        differential_diagnosis: List[Dict],
    ) -> ExplanationResult:
        """Parse JSON from LLM response."""
        try:
            data = json.loads(raw_response.strip())
            
            return ExplanationResult(
                explanation=data.get("explanation", ""),
                urgency=data.get("urgency", "routine"),
                next_steps=data.get("next_steps", ["Consult a healthcare professional"]),
                raw_response=raw_response,
            )
        except json.JSONDecodeError as e:
            logger.warning("json_parse_failed", error=str(e)[:50])
            return self._fallback_explanation(specialty, differential_diagnosis)
    
    def _fallback_explanation(
        self,
        specialty: str,
        differential_diagnosis: List[Dict],
    ) -> ExplanationResult:
        """Generate fallback explanation when LLM fails."""
        specialty_name = specialty.replace("_", " ").title()
        
        urgency_map = {
            "emergency": ("emergency", "Call 911 or go to emergency room immediately"),
            "cardiology": ("urgent", "See a cardiologist within 24-48 hours"),
            "pulmonology": ("urgent", "See a pulmonologist within 24-48 hours"),
            "neurology": ("urgent", "See a neurologist within 24-48 hours"),
            "gastroenterology": ("routine", "Schedule appointment with gastroenterologist"),
            "general_medicine": ("routine", "Schedule appointment with your doctor"),
        }
        
        urgency, step = urgency_map.get(specialty, ("routine", "Consult a healthcare professional"))
        
        top_condition = ""
        if differential_diagnosis:
            top_condition = f" The most likely condition is {differential_diagnosis[0]['condition']}."
        
        return ExplanationResult(
            explanation=f"Based on your symptoms, you should see a {specialty_name} specialist.{top_condition} Please consult a healthcare professional for proper evaluation.",
            urgency=urgency,
            next_steps=[step, "Bring a list of your symptoms and medications"],
        )
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close HTTP client."""
        self._client.close()


_generator: Optional[ExplanationGenerator] = None


def get_explanation_generator() -> ExplanationGenerator:
    """Get or create singleton generator."""
    global _generator
    if _generator is None:
        _generator = ExplanationGenerator()
    return _generator
