"""
LLM-based routing using chain-of-thought reasoning.
Uses abstracted LLM provider (Ollama/Claude/OpenAI).
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import structlog

from app.core.llm_provider import get_llm_provider, LLMProvider

logger = structlog.get_logger(__name__)

SPECIALTIES = [
    "cardiology",
    "pulmonology", 
    "neurology",
    "gastroenterology",
    "orthopedics",
    "dermatology",
    "emergency",
    "general_medicine",
]

SYSTEM_PROMPT = """You are a medical triage assistant. Your job is to route patients to the correct medical specialty based on their symptoms.

Available specialties:
- cardiology: Heart and cardiovascular issues
- pulmonology: Lung and respiratory issues
- neurology: Brain, nerves, headaches, seizures
- gastroenterology: Digestive system, stomach, intestines
- orthopedics: Bones, joints, muscles
- dermatology: Skin conditions
- emergency: Life-threatening or urgent conditions
- general_medicine: General health issues, unclear cases

You must respond with valid JSON only. No other text."""

ROUTING_PROMPT = """Analyze these patient symptoms and route to the appropriate specialty.

Patient Information:
- Symptoms: {symptoms}
- Age: {age}
- Sex: {sex}
- Medical History: {medical_history}
- Duration: {duration}
- Severity: {severity}

Think step by step:
1. What are the key symptoms?
2. What conditions could these indicate?
3. Which specialty handles these conditions?
4. Is this urgent/emergency?

Respond with this exact JSON format:
{{
    "reasoning": ["step 1 reasoning", "step 2 reasoning", "step 3 reasoning"],
    "possible_conditions": ["condition1", "condition2"],
    "primary_specialty": "specialty_name",
    "secondary_specialty": "specialty_name_or_null",
    "confidence": 0.85,
    "urgency": "emergency|urgent|routine",
    "recommendations": ["recommendation1", "recommendation2"]
}}"""


@dataclass
class LLMRoutingResult:
    """Result from LLM-based routing."""

    primary_specialty: str
    secondary_specialty: Optional[str]
    confidence: float
    urgency: str
    reasoning: List[str]
    possible_conditions: List[str]
    recommendations: List[str]
    raw_response: str


class LLMRouter:
    """Route symptoms to specialties using LLM reasoning."""

    def __init__(self) -> None:
        """Initialize with LLM provider."""
        self._provider: Optional[LLMProvider] = None

    def _get_provider(self) -> LLMProvider:
        """Lazy load LLM provider."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    async def route(
        self,
        symptoms: List[str],
        age: Optional[int] = None,
        sex: Optional[str] = None,
        medical_history: Optional[List[str]] = None,
        duration: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> LLMRoutingResult:
        """
        Route symptoms using LLM with chain-of-thought reasoning.

        Args:
            symptoms: List of extracted symptom names
            age: Patient age
            sex: Patient sex
            medical_history: List of conditions/history
            duration: Symptom duration
            severity: Symptom severity

        Returns:
            LLMRoutingResult with specialty, confidence, reasoning
        """
        provider = self._get_provider()

        # Format prompt
        prompt = ROUTING_PROMPT.format(
            symptoms=", ".join(symptoms) if symptoms else "not specified",
            age=age if age else "not specified",
            sex=sex if sex else "not specified",
            medical_history=", ".join(medical_history) if medical_history else "none",
            duration=duration if duration else "not specified",
            severity=severity if severity else "not specified",
        )

        try:
            # Get LLM response
            raw_response = await provider.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=1024,
            )

            # Parse JSON from response
            result = self._parse_response(raw_response)

            logger.info(
                "llm_routing_complete",
                primary_specialty=result.primary_specialty,
                confidence=result.confidence,
                urgency=result.urgency,
            )

            return result

        except Exception as e:
            logger.error("llm_routing_failed", error=str(e))
            # Return fallback result
            return LLMRoutingResult(
                primary_specialty="general_medicine",
                secondary_specialty=None,
                confidence=0.3,
                urgency="routine",
                reasoning=[f"LLM routing failed: {str(e)}"],
                possible_conditions=[],
                recommendations=["Consult general medicine for evaluation"],
                raw_response="",
            )

    def _parse_response(self, raw_response: str) -> LLMRoutingResult:
        """Parse LLM response into structured result."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if not json_match:
            raise ValueError("No JSON found in response")

        json_str = json_match.group(0)
        data = json.loads(json_str)

        # Validate and normalize specialty
        primary = data.get("primary_specialty", "general_medicine").lower()
        if primary not in SPECIALTIES:
            primary = "general_medicine"

        secondary = data.get("secondary_specialty")
        if secondary:
            secondary = secondary.lower()
            if secondary not in SPECIALTIES or secondary == "null":
                secondary = None

        # Validate urgency
        urgency = data.get("urgency", "routine").lower()
        if urgency not in ["emergency", "urgent", "routine"]:
            urgency = "routine"

        # Validate confidence
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return LLMRoutingResult(
            primary_specialty=primary,
            secondary_specialty=secondary,
            confidence=confidence,
            urgency=urgency,
            reasoning=data.get("reasoning", []),
            possible_conditions=data.get("possible_conditions", []),
            recommendations=data.get("recommendations", []),
            raw_response=raw_response,
        )


# Singleton instance
llm_router = LLMRouter()
