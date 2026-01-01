"""
Knowledge Graph router using Neo4j.
Routes symptoms to medical specialties via disease relationships.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class SpecialtyScore:
    """Score for a specialty based on KG routing."""

    specialty: str
    score: float
    matched_symptoms: List[str]
    reasoning: List[str]


@dataclass
class KGRoutingResult:
    """Result from knowledge graph routing."""

    primary_specialty: Optional[str]
    specialty_scores: Dict[str, SpecialtyScore]
    confidence: float


class KnowledgeGraphRouter:
    """Route symptoms to specialties using Neo4j knowledge graph."""

    def __init__(self) -> None:
        """Initialize Neo4j connection."""
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            logger.info("neo4j_connected", uri=settings.neo4j_uri)

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j_disconnected")

    async def health_check(self) -> bool:
        """Check if Neo4j is accessible."""
        try:
            await self.connect()
            async with self._driver.session() as session:
                result = await session.run("RETURN 1 AS health")
                await result.single()
            return True
        except Exception as e:
            logger.error("neo4j_health_check_failed", error=str(e))
            return False

    async def route(self, symptoms: List[str]) -> KGRoutingResult:
        """
        Route symptoms to specialties using knowledge graph.

        Args:
            symptoms: List of canonical symptom names

        Returns:
            KGRoutingResult with specialty scores and confidence
        """
        if not symptoms:
            return KGRoutingResult(
                primary_specialty=None,
                specialty_scores={},
                confidence=0.0,
            )

        await self.connect()

        specialty_scores: Dict[str, SpecialtyScore] = {}

        async with self._driver.session() as session:
            # Query for direct symptom → specialty relationships
            direct_query = """
            MATCH (s:Symptom)-[r:INDICATES]->(spec:Specialty)
            WHERE s.name IN $symptoms
            RETURN spec.name AS specialty, 
                   SUM(r.weight) AS total_weight,
                   COLLECT(s.name) AS matched_symptoms
            ORDER BY total_weight DESC
            """

            result = await session.run(direct_query, symptoms=symptoms)
            records = await result.data()

            for record in records:
                specialty = record["specialty"]
                weight = record["total_weight"]
                matched = record["matched_symptoms"]

                specialty_scores[specialty] = SpecialtyScore(
                    specialty=specialty,
                    score=weight,
                    matched_symptoms=matched,
                    reasoning=[
                        f"Symptoms {matched} directly indicate {specialty}"
                    ],
                )

            # Query for symptom → disease → specialty relationships
            disease_query = """
            MATCH (s:Symptom)-[r1:SYMPTOM_OF]->(d:Disease)-[r2:TREATED_BY]->(spec:Specialty)
            WHERE s.name IN $symptoms
            RETURN spec.name AS specialty,
                   SUM(r1.weight * r2.weight) AS total_weight,
                   COLLECT(DISTINCT s.name) AS matched_symptoms,
                   COLLECT(DISTINCT d.name) AS diseases
            ORDER BY total_weight DESC
            """

            result = await session.run(disease_query, symptoms=symptoms)
            records = await result.data()

            for record in records:
                specialty = record["specialty"]
                weight = record["total_weight"]
                matched = record["matched_symptoms"]
                diseases = record["diseases"]

                if specialty in specialty_scores:
                    # Add to existing score
                    specialty_scores[specialty].score += weight
                    specialty_scores[specialty].reasoning.append(
                        f"Diseases {diseases} also suggest {specialty}"
                    )
                else:
                    specialty_scores[specialty] = SpecialtyScore(
                        specialty=specialty,
                        score=weight,
                        matched_symptoms=matched,
                        reasoning=[
                            f"Symptoms suggest {diseases} → {specialty}"
                        ],
                    )

        # Normalize scores and find primary specialty
        if specialty_scores:
            max_score = max(s.score for s in specialty_scores.values())
            for spec in specialty_scores.values():
                spec.score = spec.score / max_score if max_score > 0 else 0.0

            primary = max(specialty_scores.items(), key=lambda x: x[1].score)
            confidence = primary[1].score

            logger.info(
                "kg_routing_complete",
                symptom_count=len(symptoms),
                primary_specialty=primary[0],
                confidence=confidence,
            )

            return KGRoutingResult(
                primary_specialty=primary[0],
                specialty_scores=specialty_scores,
                confidence=confidence,
            )

        return KGRoutingResult(
            primary_specialty=None,
            specialty_scores={},
            confidence=0.0,
        )


# Singleton instance
kg_router = KnowledgeGraphRouter()
