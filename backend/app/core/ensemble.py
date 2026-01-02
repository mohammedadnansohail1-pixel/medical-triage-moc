"""
Ensemble router combining Classifier, Knowledge Graph, and LLM routing.
Weighted combination with emergency overrides.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import structlog

from app.core.entity_extractor import ExtractedEntities
from app.core.knowledge_graph import kg_router, KGRoutingResult
from app.core.llm_router import llm_router, LLMRoutingResult
from app.core.classifier.router import classifier_router, ClassifierResult

logger = structlog.get_logger(__name__)

# Updated ensemble weights
WEIGHTS = {
    "classifier": 0.40,
    "llm": 0.40,
    "knowledge_graph": 0.20,
}

# Emergency symptoms - override to emergency specialty
EMERGENCY_SYMPTOMS = {
    "chest_pain": {"with": ["shortness_of_breath", "sweating"], "urgency": "emergency"},
    "loss_of_consciousness": {"with": [], "urgency": "emergency"},
    "seizure": {"with": [], "urgency": "emergency"},
    "confusion": {"with": ["fever"], "urgency": "emergency"},
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
        Route using ensemble of Classifier, LLM, and KG.
        """
        symptoms = entities.symptoms
        reasoning = []
        component_scores: Dict[str, Dict[str, float]] = {}

        # 1. Check emergency rules first (override)
        emergency_check = self._check_emergency_rules(symptoms, entities.severity)
        if emergency_check["is_emergency"]:
            reasoning.append(f"Emergency override: {emergency_check['reason']}")

        # 2. Classifier routing (NEW - 40%)
        classifier_result: Optional[ClassifierResult] = None
        try:
            classifier_result = classifier_router.route(
                symptoms=symptoms,
                age=age,
                sex=sex,
            )
            component_scores["classifier"] = classifier_result.all_probabilities
            reasoning.append(
                f"Classifier: {classifier_result.primary_specialty} "
                f"(confidence: {classifier_result.confidence:.2f})"
            )
        except Exception as e:
            logger.warning("classifier_routing_skipped", error=str(e))
            reasoning.append(f"Classifier unavailable: {str(e)[:50]}")

        # 3. Knowledge Graph routing (20%)
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

        # 4. LLM routing (40%)
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
            reasoning.append(f"LLM routing unavailable: {str(e)[:50]}")

        # 5. Combine scores with weights
        final_scores = self._combine_scores(component_scores)

        # 6. Determine final specialty
        if emergency_check["is_emergency"]:
            primary_specialty = "emergency"
            urgency = "emergency"
            secondary_specialty = None
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

        # Calculate confidence from final scores
        confidence = final_scores.get(primary_specialty, 0.5) if final_scores else 0.3

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

        # Severe + critical symptoms (but less aggressive than before)
        if severity == "severe" and any(
            s in symptoms for s in ["chest_pain", "loss_of_consciousness"]
        ):
            return {
                "is_emergency": True,
                "reason": "Severe symptoms requiring emergency evaluation",
            }

        return {"is_emergency": False, "reason": None}

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

        # Normalize
        if final_scores:
            max_score = max(final_scores.values())
            if max_score > 0:
                final_scores = {k: v / max_score for k, v in final_scores.items()}

        return final_scores


# Singleton instance
ensemble_router = EnsembleRouter()
