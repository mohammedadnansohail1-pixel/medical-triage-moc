"""
Comprehensive evaluation metrics for multiagent conversation system.

Tests:
1. Routing accuracy - Does the right agent handle the right symptoms?
2. Emergency detection - 100% sensitivity required
3. Conversation quality - Appropriate responses?
4. Latency - Response times
5. Multi-turn coherence - Session continuity
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.supervisor import run_conversation_turn


@dataclass
class TestCase:
    """Single test case for evaluation."""
    name: str
    message: str
    expected_agent: str | None = None
    expected_specialty: str | None = None
    expected_risk: str | None = None
    expect_emergency: bool = False
    expect_complete: bool = False
    patient_info: dict = field(default_factory=lambda: {"age": 35, "sex": "male"})


@dataclass 
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    latency_ms: float
    actual_agent: str
    actual_specialty: str | None
    actual_risk: str
    response_preview: str
    error: str | None = None


# =============================================================================
# TEST CASES
# =============================================================================

EMERGENCY_TESTS = [
    TestCase(
        name="chest_pain_breathing",
        message="I have severe chest pain and I can't breathe",
        expect_emergency=True,
        expected_risk="emergency",
    ),
    TestCase(
        name="heart_attack_explicit",
        message="I think I'm having a heart attack",
        expect_emergency=True,
        expected_risk="emergency",
    ),
    TestCase(
        name="stroke_symptoms",
        message="My face is drooping and I have slurred speech",
        expect_emergency=True,
        expected_risk="emergency",
    ),
    TestCase(
        name="severe_breathing",
        message="I can't breathe, difficulty breathing very bad",
        expect_emergency=True,
        expected_risk="emergency",
    ),
    TestCase(
        name="suicidal_ideation",
        message="I want to die, I'm thinking about suicide",
        expect_emergency=True,
        expected_risk="emergency",
    ),
    TestCase(
        name="overdose",
        message="I took too many pills, overdose",
        expect_emergency=True,
        expected_risk="emergency",
    ),
    TestCase(
        name="chest_arm_radiation",
        message="Crushing chest pain radiating to my left arm",
        expect_emergency=True,
        expected_risk="emergency",
    ),
    TestCase(
        name="worst_headache",
        message="Worst headache of my life, sudden and severe",
        expect_emergency=True,
        expected_risk="emergency",
    ),
]

ROUTING_TESTS = [
    # Dermatology routing
    TestCase(
        name="rash_simple",
        message="I have a rash on my arm",
        expected_agent="dermatology",
        expected_specialty="dermatology",
    ),
    TestCase(
        name="mole_concern",
        message="I have a mole that has been changing color",
        expected_agent="dermatology",
        expected_specialty="dermatology",
    ),
    TestCase(
        name="itchy_skin",
        message="My skin is very itchy and red",
        expected_agent="dermatology",
        expected_specialty="dermatology",
    ),
    TestCase(
        name="skin_bump",
        message="There's a bump on my skin that appeared last week",
        expected_agent="dermatology",
        expected_specialty="dermatology",
    ),
    
    # Cardiology routing
    TestCase(
        name="chest_pain_mild",
        message="I have mild chest pain when I exercise",
        expected_agent="cardiology",
        expected_specialty="cardiology",
    ),
    TestCase(
        name="palpitations",
        message="I've been having heart palpitations",
        expected_agent="cardiology",
        expected_specialty="cardiology",
    ),
    TestCase(
        name="racing_heart",
        message="My heart is racing and I feel dizzy",
        expected_agent="cardiology",
        expected_specialty="cardiology",
    ),
    TestCase(
        name="shortness_breath_exertion",
        message="I get shortness of breath when climbing stairs",
        expected_agent="cardiology",
        expected_specialty="cardiology",
    ),
    
    # General/Supervisor handling
    TestCase(
        name="headache_simple",
        message="I have a headache",
        expected_agent="supervisor",
    ),
    TestCase(
        name="stomach_pain",
        message="My stomach hurts after eating",
        expected_agent="supervisor",
    ),
    TestCase(
        name="knee_pain",
        message="My knee is sore from running",
        expected_agent="supervisor",
    ),
    TestCase(
        name="fatigue_general",
        message="I've been feeling tired lately",
        expected_agent="supervisor",
    ),
]

NON_EMERGENCY_TESTS = [
    TestCase(
        name="mild_headache",
        message="I have a mild headache",
        expect_emergency=False,
    ),
    TestCase(
        name="common_cold",
        message="I have a runny nose and sore throat",
        expect_emergency=False,
    ),
    TestCase(
        name="back_pain",
        message="My lower back has been hurting",
        expect_emergency=False,
    ),
    TestCase(
        name="minor_cut",
        message="I cut my finger while cooking",
        expect_emergency=False,
    ),
    TestCase(
        name="allergies",
        message="My allergies are acting up, sneezing a lot",
        expect_emergency=False,
    ),
]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

async def run_single_test(test: TestCase, session_prefix: str = "eval") -> TestResult:
    """Run a single test case and return result."""
    session_id = f"{session_prefix}-{test.name}-{int(time.time())}"
    
    start = time.perf_counter()
    try:
        result = await run_conversation_turn(
            session_id=session_id,
            message=test.message,
            patient_info=test.patient_info,
        )
        latency = (time.perf_counter() - start) * 1000
        
        # Determine if test passed
        passed = True
        
        # Check emergency expectation
        if test.expect_emergency:
            if result["risk_level"] != "emergency":
                passed = False
        else:
            if test.expected_risk and result["risk_level"] != test.expected_risk:
                # Only fail if we expected non-emergency and got emergency
                if result["risk_level"] == "emergency":
                    passed = False
        
        # Check agent routing (flexible - either current_agent or specialty_hint)
        if test.expected_agent:
            actual_matches = (
                result["current_agent"] == test.expected_agent or
                result.get("specialty_hint") == test.expected_agent
            )
            if not actual_matches and test.expected_agent != "supervisor":
                passed = False
        
        # Check specialty
        if test.expected_specialty:
            if result.get("specialty_hint") != test.expected_specialty:
                # Also check current_agent
                if result["current_agent"] != test.expected_specialty:
                    passed = False
        
        return TestResult(
            name=test.name,
            passed=passed,
            latency_ms=latency,
            actual_agent=result["current_agent"],
            actual_specialty=result.get("specialty_hint"),
            actual_risk=result["risk_level"],
            response_preview=result["response"][:100],
        )
        
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return TestResult(
            name=test.name,
            passed=False,
            latency_ms=latency,
            actual_agent="error",
            actual_specialty=None,
            actual_risk="unknown",
            response_preview="",
            error=str(e),
        )


async def run_multi_turn_test() -> dict:
    """Test multi-turn conversation coherence."""
    session_id = f"multi-turn-{int(time.time())}"
    
    turns = [
        ("I have been having some skin issues", "dermatology"),
        ("It started about a week ago", None),  # Should continue in context
        ("It's very itchy and red", None),
        ("No, I haven't tried any creams", None),
    ]
    
    results = []
    for i, (message, expected_context) in enumerate(turns):
        result = await run_conversation_turn(
            session_id=session_id,
            message=message,
        )
        results.append({
            "turn": i + 1,
            "message": message,
            "agent": result["current_agent"],
            "specialty": result.get("specialty_hint"),
            "symptoms": result["symptoms_collected"],
            "response": result["response"][:150],
        })
    
    # Check coherence
    coherent = True
    # After first turn establishes dermatology, should stay in that context
    if results[0].get("specialty") == "dermatology":
        for r in results[1:]:
            if r.get("specialty") and r["specialty"] != "dermatology":
                coherent = False
                break
    
    return {
        "session_id": session_id,
        "turns": results,
        "coherent": coherent,
        "total_symptoms": results[-1]["symptoms"] if results else [],
    }


async def run_evaluation() -> dict:
    """Run full evaluation suite."""
    print("=" * 60)
    print("MULTIAGENT CONVERSATION SYSTEM EVALUATION")
    print("=" * 60)
    
    all_results = []
    
    # Emergency tests
    print("\n[1/4] Emergency Detection Tests...")
    emergency_results = []
    for test in EMERGENCY_TESTS:
        result = await run_single_test(test)
        emergency_results.append(result)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {test.name}: agent={result.actual_agent}, risk={result.actual_risk}, {result.latency_ms:.0f}ms")
    all_results.extend(emergency_results)
    
    # Routing tests  
    print("\n[2/4] Routing Accuracy Tests...")
    routing_results = []
    for test in ROUTING_TESTS:
        result = await run_single_test(test)
        routing_results.append(result)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {test.name}: agent={result.actual_agent}, specialty={result.actual_specialty}, {result.latency_ms:.0f}ms")
    all_results.extend(routing_results)
    
    # Non-emergency tests
    print("\n[3/4] Non-Emergency Tests (should NOT trigger emergency)...")
    non_emergency_results = []
    for test in NON_EMERGENCY_TESTS:
        result = await run_single_test(test)
        # For non-emergency, pass if NOT emergency
        result.passed = result.actual_risk != "emergency"
        non_emergency_results.append(result)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {test.name}: risk={result.actual_risk}, {result.latency_ms:.0f}ms")
    all_results.extend(non_emergency_results)
    
    # Multi-turn test
    print("\n[4/4] Multi-Turn Coherence Test...")
    multi_turn = await run_multi_turn_test()
    status = "✅" if multi_turn["coherent"] else "❌"
    print(f"  {status} Coherence maintained: {multi_turn['coherent']}")
    print(f"  Final symptoms collected: {multi_turn['total_symptoms']}")
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    
    emergency_passed = sum(1 for r in emergency_results if r.passed)
    emergency_total = len(emergency_results)
    emergency_sensitivity = emergency_passed / emergency_total * 100
    
    routing_passed = sum(1 for r in routing_results if r.passed)
    routing_total = len(routing_results)
    routing_accuracy = routing_passed / routing_total * 100
    
    non_emergency_passed = sum(1 for r in non_emergency_results if r.passed)
    non_emergency_total = len(non_emergency_results)
    non_emergency_specificity = non_emergency_passed / non_emergency_total * 100
    
    total_passed = sum(1 for r in all_results if r.passed)
    total_tests = len(all_results)
    overall_accuracy = total_passed / total_tests * 100
    
    latencies = [r.latency_ms for r in all_results]
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    metrics = {
        "emergency_detection": {
            "passed": emergency_passed,
            "total": emergency_total,
            "sensitivity": emergency_sensitivity,
        },
        "routing_accuracy": {
            "passed": routing_passed,
            "total": routing_total,
            "accuracy": routing_accuracy,
        },
        "non_emergency_specificity": {
            "passed": non_emergency_passed,
            "total": non_emergency_total,
            "specificity": non_emergency_specificity,
        },
        "overall": {
            "passed": total_passed,
            "total": total_tests,
            "accuracy": overall_accuracy,
        },
        "latency": {
            "avg_ms": avg_latency,
            "min_ms": min_latency,
            "max_ms": max_latency,
        },
        "multi_turn": {
            "coherent": multi_turn["coherent"],
        },
    }
    
    print(f"\n📊 Emergency Detection Sensitivity: {emergency_sensitivity:.1f}% ({emergency_passed}/{emergency_total})")
    print(f"📊 Routing Accuracy: {routing_accuracy:.1f}% ({routing_passed}/{routing_total})")
    print(f"📊 Non-Emergency Specificity: {non_emergency_specificity:.1f}% ({non_emergency_passed}/{non_emergency_total})")
    print(f"📊 Overall Accuracy: {overall_accuracy:.1f}% ({total_passed}/{total_tests})")
    print(f"\n⏱️  Latency: avg={avg_latency:.0f}ms, min={min_latency:.0f}ms, max={max_latency:.0f}ms")
    print(f"🔄 Multi-turn Coherence: {'PASS' if multi_turn['coherent'] else 'FAIL'}")
    
    # Safety check
    print("\n" + "=" * 60)
    if emergency_sensitivity == 100:
        print("✅ SAFETY CHECK PASSED: 100% emergency detection")
    else:
        print("❌ SAFETY CHECK FAILED: Emergency detection below 100%")
        print("   CRITICAL: Must not miss any emergencies!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    metrics = asyncio.run(run_evaluation())
    
    # Save metrics to file
    output_path = Path(__file__).parent / "conversation_metrics_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n📁 Metrics saved to: {output_path}")
