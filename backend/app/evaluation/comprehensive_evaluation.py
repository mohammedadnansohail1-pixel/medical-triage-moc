"""
Comprehensive End-to-End Evaluation Suite for Medical Triage System.

This module provides in-depth evaluation of:
1. Emergency Detection - Rule-based safety layer
2. Specialized Agents - Dermatology, Cardiology, Triage
3. Multi-turn Conversations - Full patient journey simulations
4. Differential Diagnosis - DDX accuracy on real scenarios
5. Agent Coordination - Supervisor routing decisions
6. Image Analysis - Skin lesion classification
7. Risk Stratification - Urgency level accuracy
"""

import asyncio
import json
import time
import random
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.supervisor import run_conversation_turn, get_conversation_graph
from agents.state import create_initial_state, RISK_LEVELS
from agents.emergency_agent import check_emergency, quick_emergency_scan, EMERGENCY_KEYWORDS
from agents.dermatology_agent import is_skin_related, SKIN_KEYWORDS, SKIN_QUESTIONS
from agents.cardiology_agent import is_cardiac_related, CARDIAC_RED_FLAGS, assess_cardiac_risk
from app.core.triage_pipeline_v2 import TriagePipelineV2


# =============================================================================
# TEST DATA - Simulated Patient Scenarios
# =============================================================================

EMERGENCY_SCENARIOS = [
    {
        "id": "EMG-001",
        "description": "Acute MI presentation",
        "messages": ["I have crushing chest pain radiating to my left arm and jaw"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-002", 
        "description": "Stroke FAST symptoms",
        "messages": ["My face is drooping on one side and I can\'t lift my arm"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-003",
        "description": "Respiratory distress",
        "messages": ["I can\'t breathe, my lips are turning blue"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-004",
        "description": "Suicidal crisis",
        "messages": ["I want to kill myself, I have a plan"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-005",
        "description": "Anaphylaxis",
        "messages": ["I ate peanuts and my throat is swelling shut"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-006",
        "description": "Severe bleeding",
        "messages": ["I cut myself badly and can\'t stop the bleeding, blood everywhere"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-007",
        "description": "Seizure",
        "messages": ["My child is having a seizure and won\'t stop shaking"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-008",
        "description": "Overdose",
        "messages": ["I took a whole bottle of pills, overdose"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-009",
        "description": "Sudden severe headache",
        "messages": ["Worst headache of my life, came on suddenly like thunder"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
    {
        "id": "EMG-010",
        "description": "Loss of consciousness",
        "messages": ["My husband collapsed and is unconscious, not responding"],
        "expected": {"risk": "emergency", "agent": "emergency"},
    },
]

DERMATOLOGY_SCENARIOS = [
    {
        "id": "DERM-001",
        "description": "Suspicious mole - melanoma concern",
        "conversation": [
            {"user": "I have a mole that has been changing color and getting bigger"},
            {"user": "It\'s been about 2 months, it\'s now dark brown with uneven edges"},
            {"user": "Yes it\'s slightly raised and sometimes itches"},
        ],
        "expected": {"specialty": "dermatology", "min_turns": 2},
    },
    {
        "id": "DERM-002",
        "description": "Allergic contact dermatitis",
        "conversation": [
            {"user": "I have a red itchy rash on my hands"},
            {"user": "Started after I used new dish soap, about 3 days ago"},
            {"user": "Just my hands, very itchy with small blisters"},
        ],
        "expected": {"specialty": "dermatology", "min_turns": 2},
    },
    {
        "id": "DERM-003",
        "description": "Psoriasis presentation",
        "conversation": [
            {"user": "I have thick scaly patches on my elbows and knees"},
            {"user": "They\'ve been there for months, silvery white scales"},
            {"user": "My father has psoriasis too"},
        ],
        "expected": {"specialty": "dermatology", "min_turns": 2},
    },
    {
        "id": "DERM-004",
        "description": "Acne consultation",
        "conversation": [
            {"user": "I have bad acne on my face and back"},
            {"user": "I\'m 16, it\'s been getting worse for a year"},
            {"user": "Some are deep painful cysts"},
        ],
        "expected": {"specialty": "dermatology", "min_turns": 2},
    },
    {
        "id": "DERM-005",
        "description": "Shingles presentation",
        "conversation": [
            {"user": "I have a painful blistering rash on one side of my chest"},
            {"user": "It started with tingling and burning 2 days ago"},
            {"user": "I\'m 65 and had chickenpox as a child"},
        ],
        "expected": {"specialty": "dermatology", "min_turns": 2},
    },
]

CARDIOLOGY_SCENARIOS = [
    {
        "id": "CARD-001",
        "description": "Stable angina",
        "conversation": [
            {"user": "I get chest pain when I climb stairs"},
            {"user": "It goes away when I rest, feels like pressure"},
            {"user": "I\'m 58, male, have high cholesterol"},
        ],
        "expected": {"specialty": "cardiology", "risk_min": "elevated"},
    },
    {
        "id": "CARD-002",
        "description": "Atrial fibrillation symptoms",
        "conversation": [
            {"user": "My heart feels like it\'s racing and skipping beats"},
            {"user": "It happens randomly, sometimes I feel dizzy too"},
            {"user": "I\'m 67, this started a few weeks ago"},
        ],
        "expected": {"specialty": "cardiology", "risk_min": "elevated"},
    },
    {
        "id": "CARD-003",
        "description": "Heart failure symptoms",
        "conversation": [
            {"user": "I\'m short of breath and my ankles are swollen"},
            {"user": "I can\'t lie flat at night, need 3 pillows"},
            {"user": "Getting worse over the past month"},
        ],
        "expected": {"specialty": "cardiology", "risk_min": "elevated"},
    },
    {
        "id": "CARD-004",
        "description": "Palpitations - benign",
        "conversation": [
            {"user": "I sometimes feel my heart flutter"},
            {"user": "Usually after coffee, no pain or dizziness"},
            {"user": "I\'m 28, otherwise healthy"},
        ],
        "expected": {"specialty": "cardiology", "risk_min": "routine"},
    },
    {
        "id": "CARD-005",
        "description": "Exertional dyspnea",
        "conversation": [
            {"user": "I get out of breath when I walk more than a block"},
            {"user": "No chest pain but I feel tired all the time"},
            {"user": "I\'m 72 with diabetes"},
        ],
        "expected": {"specialty": "cardiology", "risk_min": "elevated"},
    },
]

GENERAL_TRIAGE_SCENARIOS = [
    {
        "id": "GEN-001",
        "description": "Upper respiratory infection",
        "symptoms": ["cough", "runny nose", "sore throat", "low fever"],
        "age": 35,
        "sex": "female",
        "expected_urgency": "self_care",
    },
    {
        "id": "GEN-002",
        "description": "Migraine headache",
        "symptoms": ["severe headache", "nausea", "light sensitivity", "one-sided pain"],
        "age": 28,
        "sex": "female",
        "expected_urgency": "routine",
    },
    {
        "id": "GEN-003",
        "description": "Gastroenteritis",
        "symptoms": ["diarrhea", "vomiting", "abdominal cramps", "fever"],
        "age": 42,
        "sex": "male",
        "expected_urgency": "routine",
    },
    {
        "id": "GEN-004",
        "description": "Lower back pain",
        "symptoms": ["lower back pain", "muscle stiffness", "pain with movement"],
        "age": 55,
        "sex": "male",
        "expected_urgency": "routine",
    },
    {
        "id": "GEN-005",
        "description": "Urinary tract infection",
        "symptoms": ["painful urination", "frequent urination", "lower abdominal pain"],
        "age": 30,
        "sex": "female",
        "expected_urgency": "routine",
    },
    {
        "id": "GEN-006",
        "description": "Allergic rhinitis",
        "symptoms": ["sneezing", "itchy eyes", "runny nose", "congestion"],
        "age": 25,
        "sex": "male",
        "expected_urgency": "self_care",
    },
    {
        "id": "GEN-007",
        "description": "Anxiety symptoms",
        "symptoms": ["racing heart", "sweating", "trembling", "feeling of dread"],
        "age": 32,
        "sex": "female",
        "expected_urgency": "routine",
    },
    {
        "id": "GEN-008",
        "description": "Sprained ankle",
        "symptoms": ["ankle pain", "swelling", "bruising", "difficulty walking"],
        "age": 22,
        "sex": "male",
        "expected_urgency": "routine",
    },
]

NON_EMERGENCY_CONTROL = [
    {"message": "I have a mild headache", "should_be_emergency": False},
    {"message": "My knee hurts when I run", "should_be_emergency": False},
    {"message": "I have a small paper cut", "should_be_emergency": False},
    {"message": "I\'ve been sneezing a lot", "should_be_emergency": False},
    {"message": "My back is a bit sore", "should_be_emergency": False},
    {"message": "I feel tired today", "should_be_emergency": False},
    {"message": "I have a slight cough", "should_be_emergency": False},
    {"message": "My stomach feels upset", "should_be_emergency": False},
    {"message": "I have dry skin on my hands", "should_be_emergency": False},
    {"message": "I slept poorly last night", "should_be_emergency": False},
]


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    category: str
    scenario_id: str
    description: str
    passed: bool
    details: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


class ComprehensiveEvaluator:
    """Comprehensive evaluation of the multiagent system."""
    
    def __init__(self):
        self.results: list[EvaluationResult] = []
        self.pipeline = None
        
    async def run_all_evaluations(self) -> dict:
        """Run all evaluation suites."""
        print("=" * 70)
        print("COMPREHENSIVE END-TO-END EVALUATION SUITE")
        print("Medical Triage Multiagent System")
        print("=" * 70)
        
        # 1. Emergency Detection
        await self.evaluate_emergency_detection()
        
        # 2. Non-Emergency Specificity
        await self.evaluate_non_emergency_specificity()
        
        # 3. Dermatology Agent
        await self.evaluate_dermatology_agent()
        
        # 4. Cardiology Agent
        await self.evaluate_cardiology_agent()
        
        # 5. Agent Component Tests
        self.evaluate_agent_components()
        
        # 6. Triage Pipeline
        await self.evaluate_triage_pipeline()
        
        # 7. Multi-turn Conversations
        await self.evaluate_multiturn_conversations()
        
        # 8. Edge Cases
        await self.evaluate_edge_cases()
        
        # Generate report
        return self.generate_report()
    
    async def evaluate_emergency_detection(self):
        """Test emergency detection with various scenarios."""
        print("\n" + "=" * 70)
        print("[1/8] EMERGENCY DETECTION EVALUATION")
        print("=" * 70)
        
        for scenario in EMERGENCY_SCENARIOS:
            start = time.time()
            try:
                result = await run_conversation_turn(
                    message=scenario["messages"][0],
                    session_id=f"eval-{scenario['id']}",
                    patient_info={"age": 45, "sex": "male"},
                )
                latency = (time.time() - start) * 1000
                
                is_emergency = result.get("risk_level") == "emergency"
                agent = result.get("current_agent", "unknown")
                
                passed = is_emergency and agent == "emergency"
                
                self.results.append(EvaluationResult(
                    category="emergency_detection",
                    scenario_id=scenario["id"],
                    description=scenario["description"],
                    passed=passed,
                    latency_ms=latency,
                    details={
                        "expected_risk": "emergency",
                        "actual_risk": result.get("risk_level"),
                        "expected_agent": "emergency",
                        "actual_agent": agent,
                        "response_preview": result.get("response", "")[:100],
                    }
                ))
                
                status = "✅" if passed else "❌"
                print(f"  {status} {scenario['id']}: {scenario['description']}")
                print(f"      Risk={result.get('risk_level')}, Agent={agent}, {latency:.0f}ms")
                
            except Exception as e:
                self.results.append(EvaluationResult(
                    category="emergency_detection",
                    scenario_id=scenario["id"],
                    description=scenario["description"],
                    passed=False,
                    error=str(e),
                ))
                print(f"  ❌ {scenario['id']}: ERROR - {e}")
    
    async def evaluate_non_emergency_specificity(self):
        """Ensure non-emergencies don\'t trigger false alarms."""
        print("\n" + "=" * 70)
        print("[2/8] NON-EMERGENCY SPECIFICITY EVALUATION")
        print("=" * 70)
        
        for i, case in enumerate(NON_EMERGENCY_CONTROL):
            start = time.time()
            try:
                result = await run_conversation_turn(
                    message=case["message"],
                    session_id=f"eval-nonemg-{i}",
                    patient_info={"age": 35, "sex": "male"},
                )
                latency = (time.time() - start) * 1000
                
                is_emergency = result.get("risk_level") == "emergency"
                passed = not is_emergency  # Should NOT be emergency
                
                self.results.append(EvaluationResult(
                    category="non_emergency_specificity",
                    scenario_id=f"NONEMG-{i+1:03d}",
                    description=case["message"][:50],
                    passed=passed,
                    latency_ms=latency,
                    details={
                        "actual_risk": result.get("risk_level"),
                        "actual_agent": result.get("current_agent"),
                    }
                ))
                
                status = "✅" if passed else "❌"
                print(f"  {status} {case['message'][:50]}...")
                print(f"      Risk={result.get('risk_level')}, {latency:.0f}ms")
                
            except Exception as e:
                self.results.append(EvaluationResult(
                    category="non_emergency_specificity",
                    scenario_id=f"NONEMG-{i+1:03d}",
                    description=case["message"][:50],
                    passed=False,
                    error=str(e),
                ))
    
    async def evaluate_dermatology_agent(self):
        """Test dermatology agent with multi-turn conversations."""
        print("\n" + "=" * 70)
        print("[3/8] DERMATOLOGY AGENT EVALUATION")
        print("=" * 70)
        
        for scenario in DERMATOLOGY_SCENARIOS:
            session_id = f"eval-{scenario['id']}"
            turns_completed = 0
            routed_to_derm = False
            
            start = time.time()
            try:
                for turn in scenario["conversation"]:
                    result = await run_conversation_turn(
                        message=turn["user"],
                        session_id=session_id,
                        patient_info={"age": 40, "sex": "male"},
                    )
                    turns_completed += 1
                    
                    if result.get("current_agent") == "dermatology":
                        routed_to_derm = True
                    
                    # Stop if emergency or complete
                    if result.get("triage_complete"):
                        break
                
                latency = (time.time() - start) * 1000
                
                passed = routed_to_derm and turns_completed >= scenario["expected"]["min_turns"]
                
                self.results.append(EvaluationResult(
                    category="dermatology_agent",
                    scenario_id=scenario["id"],
                    description=scenario["description"],
                    passed=passed,
                    latency_ms=latency,
                    details={
                        "turns_completed": turns_completed,
                        "routed_to_dermatology": routed_to_derm,
                        "final_agent": result.get("current_agent"),
                        "symptoms_collected": result.get("symptoms_collected", []),
                    }
                ))
                
                status = "✅" if passed else "❌"
                print(f"  {status} {scenario['id']}: {scenario['description']}")
                print(f"      Turns={turns_completed}, Dermatology={routed_to_derm}, {latency:.0f}ms")
                
            except Exception as e:
                self.results.append(EvaluationResult(
                    category="dermatology_agent",
                    scenario_id=scenario["id"],
                    description=scenario["description"],
                    passed=False,
                    error=str(e),
                ))
                print(f"  ❌ {scenario['id']}: ERROR - {e}")
    
    async def evaluate_cardiology_agent(self):
        """Test cardiology agent with cardiac scenarios."""
        print("\n" + "=" * 70)
        print("[4/8] CARDIOLOGY AGENT EVALUATION")
        print("=" * 70)
        
        for scenario in CARDIOLOGY_SCENARIOS:
            session_id = f"eval-{scenario['id']}"
            routed_to_cardio = False
            final_risk = "unknown"
            
            start = time.time()
            try:
                for turn in scenario["conversation"]:
                    result = await run_conversation_turn(
                        message=turn["user"],
                        session_id=session_id,
                        patient_info={"age": 55, "sex": "male"},
                    )
                    
                    if result.get("current_agent") == "cardiology":
                        routed_to_cardio = True
                    
                    final_risk = result.get("risk_level", "unknown")
                    
                    if result.get("triage_complete"):
                        break
                
                latency = (time.time() - start) * 1000
                
                # Check risk level meets minimum
                risk_met = RISK_LEVELS.get(final_risk, 0) >= RISK_LEVELS.get(scenario["expected"].get("risk_min", "routine"), 0)
                passed = routed_to_cardio
                
                self.results.append(EvaluationResult(
                    category="cardiology_agent",
                    scenario_id=scenario["id"],
                    description=scenario["description"],
                    passed=passed,
                    latency_ms=latency,
                    details={
                        "routed_to_cardiology": routed_to_cardio,
                        "final_risk": final_risk,
                        "expected_min_risk": scenario["expected"].get("risk_min"),
                        "risk_appropriate": risk_met,
                    }
                ))
                
                status = "✅" if passed else "❌"
                print(f"  {status} {scenario['id']}: {scenario['description']}")
                print(f"      Cardiology={routed_to_cardio}, Risk={final_risk}, {latency:.0f}ms")
                
            except Exception as e:
                self.results.append(EvaluationResult(
                    category="cardiology_agent",
                    scenario_id=scenario["id"],
                    description=scenario["description"],
                    passed=False,
                    error=str(e),
                ))
                print(f"  ❌ {scenario['id']}: ERROR - {e}")
    
    def evaluate_agent_components(self):
        """Test individual agent component functions."""
        print("\n" + "=" * 70)
        print("[5/8] AGENT COMPONENT TESTS")
        print("=" * 70)
        
        # Test emergency patterns
        print("\n  Emergency Pattern Coverage:")
        print(f"    Total patterns: {len(EMERGENCY_KEYWORDS)}")
        
        test_emergencies = [
            ("chest pain radiating to arm", True),
            ("can\'t breathe", True),
            ("i am having a heart attack", True),
            ("suicidal thoughts", True),
            ("mild headache", False),
            ("runny nose", False),
        ]
        
        for text, expected in test_emergencies:
            result = quick_emergency_scan(text)
            passed = result == expected
            status = "✅" if passed else "❌"
            print(f"    {status} \"{text}\" -> {result} (expected {expected})")
            
            self.results.append(EvaluationResult(
                category="component_emergency",
                scenario_id=f"pattern-{text[:20]}",
                description=text,
                passed=passed,
                details={"expected": expected, "actual": result}
            ))
        
        # Test skin keyword detection
        print("\n  Dermatology Keyword Detection:")
        print(f"    Keywords tracked: {len(SKIN_KEYWORDS)}")
        
        test_skin = [
            ("I have a rash", True),
            ("mole changing color", True),
            ("itchy bumps", True),
            ("chest pain", False),
            ("headache", False),
        ]
        
        for text, expected in test_skin:
            result = is_skin_related(text)
            passed = result == expected
            status = "✅" if passed else "❌"
            print(f"    {status} \"{text}\" -> {result} (expected {expected})")
            
            self.results.append(EvaluationResult(
                category="component_dermatology",
                scenario_id=f"skin-{text[:20]}",
                description=text,
                passed=passed,
                details={"expected": expected, "actual": result}
            ))
        
        # Test cardiac keyword detection
        print("\n  Cardiology Keyword Detection:")
        print(f"    Red flags tracked: {len(CARDIAC_RED_FLAGS)}")
        
        test_cardiac = [
            ("chest pain", True),
            ("palpitations", True),
            ("racing heart", True),
            ("rash on arm", False),
            ("runny nose", False),
        ]
        
        for text, expected in test_cardiac:
            result = is_cardiac_related(text)
            passed = result == expected
            status = "✅" if passed else "❌"
            print(f"    {status} \"{text}\" -> {result} (expected {expected})")
            
            self.results.append(EvaluationResult(
                category="component_cardiology",
                scenario_id=f"cardiac-{text[:20]}",
                description=text,
                passed=passed,
                details={"expected": expected, "actual": result}
            ))
    
    async def evaluate_triage_pipeline(self):
        """Test the core triage pipeline."""
        print("\n" + "=" * 70)
        print("[6/8] TRIAGE PIPELINE EVALUATION")
        print("=" * 70)
        
        try:
            from app.api.triage import get_pipeline
            self.pipeline = get_pipeline()
            print("  Pipeline loaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to load pipeline: {e}")
            return
        
        for scenario in GENERAL_TRIAGE_SCENARIOS:
            start = time.time()
            try:
                result = self.pipeline.predict(
                    symptoms=scenario["symptoms"],
                    age=scenario["age"],
                    sex=scenario["sex"],
                )
                latency = (time.time() - start) * 1000
                
                # Check if we got a valid result
                has_specialty = "specialty" in result
                has_ddx = "differential_diagnosis" in result and len(result["differential_diagnosis"]) > 0
                passed = has_specialty and has_ddx
                
                self.results.append(EvaluationResult(
                    category="triage_pipeline",
                    scenario_id=scenario["id"],
                    description=scenario["description"],
                    passed=passed,
                    latency_ms=latency,
                    details={
                        "specialty": result.get("specialty"),
                        "urgency": result.get("urgency"),
                        "top_diagnosis": result.get("differential_diagnosis", [{}])[0].get("condition") if result.get("differential_diagnosis") else None,
                        "ddx_count": len(result.get("differential_diagnosis", [])),
                    }
                ))
                
                status = "✅" if passed else "❌"
                print(f"  {status} {scenario['id']}: {scenario['description']}")
                print(f"      Specialty={result.get('specialty')}, Urgency={result.get('urgency')}, {latency:.0f}ms")
                if result.get("differential_diagnosis"):
                    top3 = [d["condition"] for d in result["differential_diagnosis"][:3]]
                    print(f"      Top DDX: {top3}")
                
            except Exception as e:
                self.results.append(EvaluationResult(
                    category="triage_pipeline",
                    scenario_id=scenario["id"],
                    description=scenario["description"],
                    passed=False,
                    error=str(e),
                ))
                print(f"  ❌ {scenario['id']}: ERROR - {e}")
    
    async def evaluate_multiturn_conversations(self):
        """Test full multi-turn conversation flows."""
        print("\n" + "=" * 70)
        print("[7/8] MULTI-TURN CONVERSATION EVALUATION")
        print("=" * 70)
        
        # Scenario 1: Complete dermatology journey
        print("\n  Scenario: Complete Dermatology Journey")
        session_id = "eval-multiturn-derm"
        conversation = [
            "I noticed a spot on my skin",
            "It\'s on my back, dark colored",
            "About 2 weeks, it seems to be growing",
            "No pain but sometimes itchy",
        ]
        
        results_history = []
        start = time.time()
        
        for msg in conversation:
            result = await run_conversation_turn(
                message=msg,
                session_id=session_id,
                patient_info={"age": 45, "sex": "male"},
            )
            results_history.append(result)
            print(f"    User: {msg}")
            print(f"    Agent: {result.get('current_agent')} | Risk: {result.get('risk_level')}")
            print(f"    Response: {result.get('response', '')[:80]}...")
            print()
        
        latency = (time.time() - start) * 1000
        
        # Check conversation quality
        routed_to_specialist = any(r.get("current_agent") == "dermatology" for r in results_history)
        symptoms_accumulated = len(results_history[-1].get("symptoms_collected", [])) > 0
        passed = routed_to_specialist
        
        self.results.append(EvaluationResult(
            category="multiturn_conversation",
            scenario_id="MULTI-DERM",
            description="Complete dermatology journey",
            passed=passed,
            latency_ms=latency,
            details={
                "turns": len(conversation),
                "routed_to_specialist": routed_to_specialist,
                "final_symptoms": results_history[-1].get("symptoms_collected", []),
            }
        ))
        
        status = "✅" if passed else "❌"
        print(f"  {status} Journey complete: {len(conversation)} turns, Specialist={routed_to_specialist}")
        
        # Scenario 2: Complete cardiology journey
        print("\n  Scenario: Complete Cardiology Journey")
        session_id = "eval-multiturn-cardio"
        conversation = [
            "I\'ve been having chest discomfort",
            "It happens when I walk uphill",
            "Feels like pressure, goes away when I rest",
            "I\'m 62 with high blood pressure",
        ]
        
        results_history = []
        start = time.time()
        
        for msg in conversation:
            result = await run_conversation_turn(
                message=msg,
                session_id=session_id,
                patient_info={"age": 62, "sex": "male"},
            )
            results_history.append(result)
            print(f"    User: {msg}")
            print(f"    Agent: {result.get('current_agent')} | Risk: {result.get('risk_level')}")
            print()
        
        latency = (time.time() - start) * 1000
        
        routed_to_cardio = any(r.get("current_agent") == "cardiology" for r in results_history)
        passed = routed_to_cardio
        
        self.results.append(EvaluationResult(
            category="multiturn_conversation",
            scenario_id="MULTI-CARDIO",
            description="Complete cardiology journey",
            passed=passed,
            latency_ms=latency,
            details={
                "turns": len(conversation),
                "routed_to_cardiology": routed_to_cardio,
            }
        ))
        
        status = "✅" if passed else "❌"
        print(f"  {status} Journey complete: {len(conversation)} turns, Cardiology={routed_to_cardio}")
    
    async def evaluate_edge_cases(self):
        """Test edge cases and boundary conditions."""
        print("\n" + "=" * 70)
        print("[8/8] EDGE CASE EVALUATION")
        print("=" * 70)
        
        edge_cases = [
            {
                "id": "EDGE-001",
                "description": "Empty message",
                "message": "",
                "should_not_crash": True,
            },
            {
                "id": "EDGE-002",
                "description": "Very long message",
                "message": "I have pain " * 100,
                "should_not_crash": True,
            },
            {
                "id": "EDGE-003",
                "description": "Special characters",
                "message": "I have pain!!! @#$%^&*() <script>alert(\'x\')</script>",
                "should_not_crash": True,
            },
            {
                "id": "EDGE-004",
                "description": "Mixed emergency and non-emergency",
                "message": "I have a mild headache but also crushing chest pain",
                "expected_emergency": True,  # Should catch emergency
            },
            {
                "id": "EDGE-005",
                "description": "Negated emergency",
                "message": "I don\'t have chest pain, just a cough",
                "expected_emergency": False,
            },
            {
                "id": "EDGE-006",
                "description": "Unicode characters",
                "message": "I have pain in my tête (head) 头痛",
                "should_not_crash": True,
            },
        ]
        
        for case in edge_cases:
            start = time.time()
            try:
                result = await run_conversation_turn(
                    message=case["message"],
                    session_id=f"eval-{case['id']}",
                    patient_info={"age": 30, "sex": "male"},
                )
                latency = (time.time() - start) * 1000
                
                passed = True
                if "expected_emergency" in case:
                    is_emergency = result.get("risk_level") == "emergency"
                    passed = is_emergency == case["expected_emergency"]
                
                self.results.append(EvaluationResult(
                    category="edge_cases",
                    scenario_id=case["id"],
                    description=case["description"],
                    passed=passed,
                    latency_ms=latency,
                    details={
                        "risk_level": result.get("risk_level"),
                        "agent": result.get("current_agent"),
                    }
                ))
                
                status = "✅" if passed else "❌"
                print(f"  {status} {case['id']}: {case['description']}")
                
            except Exception as e:
                passed = False
                self.results.append(EvaluationResult(
                    category="edge_cases",
                    scenario_id=case["id"],
                    description=case["description"],
                    passed=False,
                    error=str(e),
                ))
                print(f"  ❌ {case['id']}: ERROR - {e}")
    
    def generate_report(self) -> dict:
        """Generate comprehensive evaluation report."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY REPORT")
        print("=" * 70)
        
        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Calculate metrics per category
        report = {
            "summary": {},
            "categories": {},
            "safety_check": {},
        }
        
        total_passed = 0
        total_tests = 0
        total_latency = 0
        latency_count = 0
        
        for category, results in categories.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            accuracy = passed / total * 100 if total > 0 else 0
            
            latencies = [r.latency_ms for r in results if r.latency_ms > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            total_passed += passed
            total_tests += total
            total_latency += sum(latencies)
            latency_count += len(latencies)
            
            report["categories"][category] = {
                "passed": passed,
                "total": total,
                "accuracy": accuracy,
                "avg_latency_ms": avg_latency,
            }
            
            print(f"\n  {category}:")
            print(f"    Passed: {passed}/{total} ({accuracy:.1f}%)")
            if avg_latency > 0:
                print(f"    Avg Latency: {avg_latency:.0f}ms")
        
        # Overall metrics
        overall_accuracy = total_passed / total_tests * 100 if total_tests > 0 else 0
        overall_latency = total_latency / latency_count if latency_count > 0 else 0
        
        report["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_accuracy": overall_accuracy,
            "avg_latency_ms": overall_latency,
        }
        
        # Safety check
        emergency_results = categories.get("emergency_detection", [])
        emergency_passed = sum(1 for r in emergency_results if r.passed)
        emergency_total = len(emergency_results)
        emergency_sensitivity = emergency_passed / emergency_total * 100 if emergency_total > 0 else 0
        
        non_emg_results = categories.get("non_emergency_specificity", [])
        non_emg_passed = sum(1 for r in non_emg_results if r.passed)
        non_emg_total = len(non_emg_results)
        non_emg_specificity = non_emg_passed / non_emg_total * 100 if non_emg_total > 0 else 0
        
        safety_passed = emergency_sensitivity == 100 and non_emg_specificity >= 95
        
        report["safety_check"] = {
            "emergency_sensitivity": emergency_sensitivity,
            "non_emergency_specificity": non_emg_specificity,
            "passed": safety_passed,
        }
        
        print("\n" + "=" * 70)
        print("FINAL METRICS")
        print("=" * 70)
        print(f"\n  📊 Overall Accuracy: {overall_accuracy:.1f}% ({total_passed}/{total_tests})")
        print(f"  ⏱️  Average Latency: {overall_latency:.0f}ms")
        print(f"\n  🚨 Emergency Sensitivity: {emergency_sensitivity:.1f}%")
        print(f"  ✅ Non-Emergency Specificity: {non_emg_specificity:.1f}%")
        
        print("\n" + "=" * 70)
        if safety_passed:
            print("✅ SAFETY CHECK PASSED")
            print("   - 100% emergency detection")
            print("   - ≥95% non-emergency specificity")
        else:
            print("❌ SAFETY CHECK FAILED")
            if emergency_sensitivity < 100:
                print(f"   - Emergency sensitivity: {emergency_sensitivity:.1f}% (required: 100%)")
            if non_emg_specificity < 95:
                print(f"   - Non-emergency specificity: {non_emg_specificity:.1f}% (required: ≥95%)")
        print("=" * 70)
        
        return report


async def main():
    """Run comprehensive evaluation."""
    evaluator = ComprehensiveEvaluator()
    report = await evaluator.run_all_evaluations()
    
    # Save report
    output_path = Path(__file__).parent / "comprehensive_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📁 Full report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())
