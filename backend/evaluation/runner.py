"""
Evaluation runner - tests system against test cases and calculates metrics.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx


@dataclass
class CaseResult:
    """Result for a single test case."""

    case_id: str
    category: str
    expected_specialty: str
    predicted_specialty: str
    expected_urgency: str
    predicted_urgency: str
    specialty_correct: bool
    urgency_correct: bool
    confidence: float
    latency_ms: float
    reasoning: List[str] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    total_cases: int
    specialty_accuracy: float
    urgency_accuracy: float
    emergency_sensitivity: float  # How many emergencies were detected
    avg_latency_ms: float
    avg_confidence: float
    per_specialty_accuracy: Dict[str, float]
    per_category_accuracy: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]


class EvaluationRunner:
    """Run evaluation against test cases."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results: List[CaseResult] = []

    async def run_single_case(
        self, case: dict, client: httpx.AsyncClient
    ) -> CaseResult:
        """Run a single test case."""
        start = time.time()

        try:
            response = await client.post(
                f"{self.api_url}/api/triage",
                json={
                    "symptoms": case["symptoms"],
                    "age": case.get("age"),
                    "sex": case.get("sex"),
                    "medical_history": case.get("medical_history", []),
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            latency = (time.time() - start) * 1000

            predicted_specialty = data["primary_specialty"]
            predicted_urgency = data["urgency"]

            # For emergency cases, both "emergency" specialty and urgency count
            expected_spec = case["expected_specialty"]
            spec_correct = predicted_specialty == expected_spec
            
            # Also accept emergency routing for emergency-urgency cases
            if case["expected_urgency"] == "emergency":
                spec_correct = spec_correct or predicted_specialty == "emergency"

            return CaseResult(
                case_id=case["id"],
                category=case["category"],
                expected_specialty=expected_spec,
                predicted_specialty=predicted_specialty,
                expected_urgency=case["expected_urgency"],
                predicted_urgency=predicted_urgency,
                specialty_correct=spec_correct,
                urgency_correct=predicted_urgency == case["expected_urgency"],
                confidence=data["confidence"],
                latency_ms=latency,
                reasoning=data.get("reasoning", []),
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return CaseResult(
                case_id=case["id"],
                category=case["category"],
                expected_specialty=case["expected_specialty"],
                predicted_specialty="error",
                expected_urgency=case["expected_urgency"],
                predicted_urgency="error",
                specialty_correct=False,
                urgency_correct=False,
                confidence=0.0,
                latency_ms=latency,
                reasoning=[f"Error: {str(e)}"],
            )

    async def run_evaluation(self, test_cases: List[dict]) -> EvaluationMetrics:
        """Run full evaluation on all test cases."""
        self.results = []

        async with httpx.AsyncClient() as client:
            for i, case in enumerate(test_cases):
                print(f"Running case {i+1}/{len(test_cases)}: {case['id']}...", end=" ")
                result = await self.run_single_case(case, client)
                self.results.append(result)
                status = "✓" if result.specialty_correct else "✗"
                print(f"{status} ({result.predicted_specialty})")

        return self.calculate_metrics()

    def calculate_metrics(self) -> EvaluationMetrics:
        """Calculate aggregate metrics from results."""
        if not self.results:
            raise ValueError("No results to calculate metrics from")

        total = len(self.results)

        # Overall accuracy
        specialty_correct = sum(1 for r in self.results if r.specialty_correct)
        urgency_correct = sum(1 for r in self.results if r.urgency_correct)

        # Emergency sensitivity
        emergency_cases = [r for r in self.results if r.expected_urgency == "emergency"]
        emergency_detected = sum(
            1 for r in emergency_cases if r.predicted_urgency == "emergency"
        )
        emergency_sensitivity = (
            emergency_detected / len(emergency_cases) if emergency_cases else 1.0
        )

        # Latency and confidence
        avg_latency = sum(r.latency_ms for r in self.results) / total
        avg_confidence = sum(r.confidence for r in self.results) / total

        # Per-specialty accuracy
        specialties = set(r.expected_specialty for r in self.results)
        per_specialty = {}
        for spec in specialties:
            spec_results = [r for r in self.results if r.expected_specialty == spec]
            if spec_results:
                correct = sum(1 for r in spec_results if r.specialty_correct)
                per_specialty[spec] = correct / len(spec_results)

        # Per-category accuracy
        categories = set(r.category for r in self.results)
        per_category = {}
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            if cat_results:
                correct = sum(1 for r in cat_results if r.specialty_correct)
                per_category[cat] = correct / len(cat_results)

        # Confusion matrix
        confusion: Dict[str, Dict[str, int]] = {}
        for r in self.results:
            expected = r.expected_specialty
            predicted = r.predicted_specialty
            if expected not in confusion:
                confusion[expected] = {}
            confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

        return EvaluationMetrics(
            total_cases=total,
            specialty_accuracy=specialty_correct / total,
            urgency_accuracy=urgency_correct / total,
            emergency_sensitivity=emergency_sensitivity,
            avg_latency_ms=avg_latency,
            avg_confidence=avg_confidence,
            per_specialty_accuracy=per_specialty,
            per_category_accuracy=per_category,
            confusion_matrix=confusion,
        )

    def print_report(self, metrics: EvaluationMetrics) -> None:
        """Print evaluation report."""
        print("\n" + "=" * 60)
        print("           MEDICAL TRIAGE EVALUATION REPORT")
        print("=" * 60)

        print(f"\n📊 OVERALL METRICS")
        print(f"   Total Cases:          {metrics.total_cases}")
        print(f"   Specialty Accuracy:   {metrics.specialty_accuracy:.1%}")
        print(f"   Urgency Accuracy:     {metrics.urgency_accuracy:.1%}")
        print(f"   Emergency Sensitivity:{metrics.emergency_sensitivity:.1%}")
        print(f"   Avg Latency:          {metrics.avg_latency_ms:.0f}ms")
        print(f"   Avg Confidence:       {metrics.avg_confidence:.2f}")

        print(f"\n📋 PER-SPECIALTY ACCURACY")
        for spec, acc in sorted(metrics.per_specialty_accuracy.items()):
            bar = "█" * int(acc * 20)
            print(f"   {spec:20} {bar:20} {acc:.1%}")

        print(f"\n🏥 PER-CATEGORY ACCURACY")
        for cat, acc in sorted(metrics.per_category_accuracy.items()):
            status = "✓" if acc >= 0.5 else "✗"
            print(f"   {status} {cat:25} {acc:.1%}")

        print(f"\n❌ MISCLASSIFIED CASES")
        for r in self.results:
            if not r.specialty_correct:
                print(f"   {r.case_id}: Expected {r.expected_specialty}, "
                      f"Got {r.predicted_specialty}")

        print("\n" + "=" * 60)


async def main():
    """Run evaluation."""
    from evaluation.test_cases import TEST_CASES

    runner = EvaluationRunner()
    metrics = await runner.run_evaluation(TEST_CASES)
    runner.print_report(metrics)

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "metrics": {
                "total_cases": metrics.total_cases,
                "specialty_accuracy": metrics.specialty_accuracy,
                "urgency_accuracy": metrics.urgency_accuracy,
                "emergency_sensitivity": metrics.emergency_sensitivity,
                "avg_latency_ms": metrics.avg_latency_ms,
                "avg_confidence": metrics.avg_confidence,
            },
            "per_specialty": metrics.per_specialty_accuracy,
            "per_category": metrics.per_category_accuracy,
            "confusion_matrix": metrics.confusion_matrix,
        }, f, indent=2)

    print("\n💾 Results saved to evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(main())
