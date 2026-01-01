"""
Ensemble router combining Knowledge Graph, LLM, and Rule-based routing.
Weighted combination with emergency overrides.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import structlog

from app.core.entity_extractor import ExtractedEntities
from app.core.knowledge_graph import kg_router, KGRoutingResult
from app.core.llm_router import llm_router, LLMRoutingResult

logger = structlog.get_logger(__name__)

# Ensemble weights
WEIGHTS = {
    "knowledge_graph": 0.30,
    "llm": 0.50,
    "rules": 0.20,
}

# Emergency symptoms - override to emergency specialty
EMERGENCY_SYMPTOMS = {
    "chest_pain": {"with": ["shortness_of_breath", "sweating"], "urgency": "emergency"},
    "loss_of_consciousness": {"with": [], "urgency": "emergency"},
    "seizure": {"with": [], "urgency": "emergency"},
    "confusion": {"with": ["fever"], "urgency": "emergency"},
    "shortness_of_breath": {"with": ["chest_pain"], "urgency": "emergency"},
}

# Severity escalation rules
SEVERITY_URGENCY_MAP = {
    "severe": "urgent",
    "moderate": "routine",
    "mild": "routine",
}


@dataclass
class EnsembleResult:
    """Final routing result from ensemble."""

    primary_specialty: str
    secondary_specialty: Optional[str]
    confidence: float
    urgency: str
    extracted_symptoms: List[str]
    reasoning: List[str]
    recommendations: List[str]
    component_scores: Dict[str, Dict[str, float]]


class EnsembleRouter:
    """Combine multiple routing signals into final decision."""

    async def route(
        self,
        entities: ExtractedEntities,
        age: Optional[int] = None,
        sex: Optional[str] = None,
        medical_history: Optional[List[str]] = None,
    ) -> EnsembleResult:
        """
        Route using ensemble of KG, LLM, and rules.

        Args:
            entities: Extracted entities from patient text
            age: Patient age
            sex: Patient sex  
            medical_history: List of medical conditions

        Returns:
            EnsembleResult with final routing decision
        """
        symptoms = entities.symptoms
        reasoning = []
        component_scores: Dict[str, Dict[str, float]] = {}

        # 1. Check emergency rules first (override)
        emergency_check = self._check_emergency_rules(symptoms, entities.severity)
        if emergency_check["is_emergency"]:
            reasoning.append(f"Emergency override: {emergency_check['reason']}")

        # 2. Knowledge Graph routing
        kg_result: Optional[KGRoutingResult] = None
        try:
            kg_result = await kg_router.route(symptoms)
            if kg_result.specialty_scores:
                component_scores["knowledge_graph"] = {
                    spec: score.score 
                    for spec, score in kg_result.specialty_scores.items()
                }
                reasoning.append(
                    f"KG routing: {kg_result.primary_specialty} "
                    f"(confidence: {kg_result.confidence:.2f})"
                )
        except Exception as e:
            logger.warning("kg_routing_skipped", error=str(e))
            reasoning.append("KG routing unavailable")

        # 3. LLM routing
        llm_result: Optional[LLMRoutingResult] = None
        try:
            llm_result = await llm_router.route(
                symptoms=symptoms,
                age=age,
                sex=sex,
                medical_history=medical_history,
                duration=entities.duration,
                severity=entities.severity,
            )
            component_scores["llm"] = {llm_result.primary_specialty: llm_result.confidence}
            if llm_result.secondary_specialty:
                component_scores["llm"][llm_result.secondary_specialty] = llm_result.confidence * 0.5
            reasoning.extend(llm_result.reasoning)
        except Exception as e:
            logger.warning("llm_routing_skipped", error=str(e))
            reasoning.append("LLM routing unavailable")

        # 4. Rule-based scoring
        rule_scores = self._apply_rules(symptoms, entities.severity, age)
        component_scores["rules"] = rule_scores
        if rule_scores:
            top_rule = max(rule_scores.items(), key=lambda x: x[1])
            reasoning.append(f"Rule-based: {top_rule[0]} (score: {top_rule[1]:.2f})")

        # 5. Combine scores with weights
        final_scores = self._combine_scores(component_scores)

        # 6. Determine final specialty
        if emergency_check["is_emergency"]:
            primary_specialty = "emergency"
            urgency = "emergency"
        elif final_scores:
            sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            primary_specialty = sorted_scores[0][0]
            secondary_specialty = sorted_scores[1][0] if len(sorted_scores) > 1 else None
            
            # Determine urgency
            if llm_result and llm_result.urgency == "emergency":
                urgency = "emergency"
            elif entities.severity == "severe":
                urgency = "urgent"
            elif llm_result:
                urgency = llm_result.urgency
            else:
                urgency = "routine"
        else:
            primary_specialty = "general_medicine"
            secondary_specialty = None
            urgency = "routine"

        # Calculate confidence
        confidence = final_scores.get(primary_specialty, 0.5) if final_scores else 0.3

        # Get secondary specialty
        if not emergency_check["is_emergency"] and final_scores:
            sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            secondary_specialty = sorted_scores[1][0] if len(sorted_scores) > 1 else None
        else:
            secondary_specialty = None

        # Compile recommendations
        recommendations = []
        if llm_result:
            recommendations.extend(llm_result.recommendations)
        if urgency == "emergency":
            recommendations.insert(0, "Seek immediate medical attention")
        elif urgency == "urgent":
            recommendations.insert(0, "Schedule appointment within 24-48 hours")

        logger.info(
            "ensemble_routing_complete",
            primary_specialty=primary_specialty,
            confidence=confidence,
            urgency=urgency,
            symptom_count=len(symptoms),
        )

        return EnsembleResult(
            primary_specialty=primary_specialty,
            secondary_specialty=secondary_specialty,
            confidence=round(confidence, 2),
            urgency=urgency,
            extracted_symptoms=symptoms,
            reasoning=reasoning,
            recommendations=recommendations,
            component_scores=component_scores,
        )

    def _check_emergency_rules(
        self, symptoms: List[str], severity: Optional[str]
    ) -> Dict:
        """Check if symptoms match emergency patterns."""
        for symptom, rule in EMERGENCY_SYMPTOMS.items():
            if symptom in symptoms:
                required_with = rule["with"]
                if not required_with or any(s in symptoms for s in required_with):
                    return {
                        "is_emergency": True,
                        "reason": f"{symptom} with severity/combination warrants emergency",
                    }

        # Severe + critical symptoms
        if severity == "severe" and any(
            s in symptoms for s in ["chest_pain", "shortness_of_breath", "confusion"]
        ):
            return {
                "is_emergency": True,
                "reason": "Severe symptoms requiring emergency evaluation",
            }

        return {"is_emergency": False, "reason": None}

    def _apply_rules(
        self,
        symptoms: List[str],
        severity: Optional[str],
        age: Optional[int],
    ) -> Dict[str, float]:
        """Apply rule-based scoring."""
        scores: Dict[str, float] = {}

        # Cardiac rules
        cardiac_symptoms = {"chest_pain", "palpitations", "shortness_of_breath"}
        cardiac_count = len(cardiac_symptoms.intersection(symptoms))
        if cardiac_count > 0:
            scores["cardiology"] = 0.3 * cardiac_count
            if age and age > 50:
                scores["cardiology"] += 0.2

        # Neuro rules
        neuro_symptoms = {"headache", "dizziness", "numbness", "confusion", "seizure"}
        neuro_count = len(neuro_symptoms.intersection(symptoms))
        if neuro_count > 0:
            scores["neurology"] = 0.3 * neuro_count

        # GI rules
        gi_symptoms = {"abdominal_pain", "nausea", "vomiting"}
        gi_count = len(gi_symptoms.intersection(symptoms))
        if gi_count > 0:
            scores["gastroenterology"] = 0.3 * gi_count

        # Pulm rules
        pulm_symptoms = {"cough", "shortness_of_breath"}
        pulm_count = len(pulm_symptoms.intersection(symptoms))
        if pulm_count > 0:
            scores["pulmonology"] = 0.3 * pulm_count

        # Normalize
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def _combine_scores(
        self, component_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine component scores with weights."""
        final_scores: Dict[str, float] = {}

        for component, weight in WEIGHTS.items():
            if component in component_scores:
                for specialty, score in component_scores[component].items():
                    if specialty not in final_scores:
                        final_scores[specialty] = 0.0
                    final_scores[specialty] += weight * score

        # Normalize final scores
        if final_scores:
            max_score = max(final_scores.values())
            if max_score > 0:
                final_scores = {k: v / max_score for k, v in final_scores.items()}

        return final_scores


# Singleton instance
ensemble_router = EnsembleRouter()
