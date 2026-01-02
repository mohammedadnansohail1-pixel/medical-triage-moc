"""
DDXPlus dataset evaluation with classifier + ensemble.
"""

import ast
import asyncio
import json
from pathlib import Path
from typing import Dict, List

import httpx
from datasets import load_dataset

from app.core.classifier.router import classifier_router


PATHOLOGY_TO_SPECIALTY: Dict[str, str] = {
    # Cardiology
    "Possible NSTEMI / STEMI": "cardiology",
    "Unstable angina": "cardiology",
    "Stable angina": "cardiology",
    "Myocarditis": "cardiology",
    "Pericarditis": "cardiology",
    "Atrial fibrillation": "cardiology",
    "SVT": "cardiology",
    "PSVT": "cardiology",
    
    # Pulmonology
    "Bronchitis": "pulmonology",
    "Pneumonia": "pulmonology",
    "COPD": "pulmonology",
    "Pulmonary embolism": "pulmonology",
    "Tuberculosis": "pulmonology",
    "Bronchiectasis": "pulmonology",
    "Acute pulmonary edema": "pulmonology",
    "Epiglottitis": "pulmonology",
    "Larygospasm": "pulmonology",
    "Acute laryngitis": "pulmonology",
    "Viral pharyngitis": "pulmonology",
    "Spontaneous pneumothorax": "pulmonology",
    "Acute COPD exacerbation / infection": "pulmonology",
    "Bronchospasm / acute asthma exacerbation": "pulmonology",
    "Croup": "pulmonology",
    "Whooping cough": "pulmonology",
    "Pulmonary neoplasm": "pulmonology",
    
    # Gastroenterology
    "GERD": "gastroenterology",
    "Boerhaave": "gastroenterology",
    "Pancreatic neoplasm": "gastroenterology",
    "Inguinal hernia": "gastroenterology",
    
    # Neurology
    "Cluster headache": "neurology",
    "Panic attack": "neurology",
    "Guillain-Barré syndrome": "neurology",
    "Myasthenia gravis": "neurology",
    
    # General Medicine
    "URTI": "general_medicine",
    "Influenza": "general_medicine",
    "Allergic sinusitis": "general_medicine",
    "Acute rhinosinusitis": "general_medicine",
    "Chronic rhinosinusitis": "general_medicine",
    "Acute otitis media": "general_medicine",
    "Chronic otitis media": "general_medicine",
    "Anemia": "general_medicine",
    "HIV (initial infection)": "general_medicine",
    "Sarcoidosis": "general_medicine",
    "Acute dystonic reactions": "general_medicine",
    "SLE": "general_medicine",
    "Chagas": "general_medicine",
    
    # Emergency
    "Anaphylaxis": "emergency",
    "Scombroid food poisoning": "emergency",
    "Spontaneous rib fracture": "emergency",
    
    # Dermatology
    "Localized edema": "dermatology",
    "Atopic dermatitis": "dermatology",
}


def get_specialty(pathology: str) -> str:
    return PATHOLOGY_TO_SPECIALTY.get(pathology, "general_medicine")


def extract_evidence_codes(evidence_str: str) -> List[str]:
    try:
        evidences = ast.literal_eval(evidence_str)
        return [ev.split("_@_")[0] for ev in evidences]
    except:
        return []


async def evaluate_classifier_only(n_samples: int = 1000) -> Dict:
    """Evaluate classifier directly on DDXPlus (no API call)."""
    print(f"Loading DDXPlus test set ({n_samples} samples)...")
    dataset = load_dataset("aai530-group6/ddxplus", split=f"test[:{n_samples}]")
    
    results = {"total": 0, "correct": 0, "by_specialty": {}}
    
    for i, case in enumerate(dataset):
        pathology = case["PATHOLOGY"]
        expected = get_specialty(pathology)
        
        if expected == "general_medicine" and pathology not in PATHOLOGY_TO_SPECIALTY:
            continue  # Skip unknown pathologies
        
        results["total"] += 1
        
        evidence_codes = extract_evidence_codes(case["EVIDENCES"])
        sex = "male" if case["SEX"] == "M" else "female"
        
        pred = classifier_router.route_from_evidences(
            evidence_codes=evidence_codes,
            age=case["AGE"],
            sex=sex,
        )
        
        is_correct = pred.primary_specialty == expected
        
        if is_correct:
            results["correct"] += 1
        
        # Track by specialty
        if expected not in results["by_specialty"]:
            results["by_specialty"][expected] = {"total": 0, "correct": 0}
        results["by_specialty"][expected]["total"] += 1
        if is_correct:
            results["by_specialty"][expected]["correct"] += 1
        
        if (i + 1) % 200 == 0:
            acc = results["correct"] / results["total"]
            print(f"  [{i+1}/{n_samples}] Accuracy: {acc:.1%}")
    
    accuracy = results["correct"] / results["total"]
    
    print(f"\n{'='*60}")
    print(f"CLASSIFIER-ONLY EVALUATION ({n_samples} samples)")
    print(f"{'='*60}")
    print(f"Accuracy: {results['correct']}/{results['total']} ({accuracy:.1%})")
    print(f"\nPer-Specialty:")
    for spec, stats in sorted(results["by_specialty"].items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {spec:20} {stats['correct']:4}/{stats['total']:4} ({acc:.0%})")
    
    return results


async def evaluate_full_api(n_samples: int = 100) -> Dict:
    """Evaluate full API (ensemble) on DDXPlus."""
    print(f"\nLoading DDXPlus test set ({n_samples} samples for API test)...")
    dataset = load_dataset("aai530-group6/ddxplus", split=f"test[:{n_samples}]")
    
    # Load evidence mapping for decoding
    evidence_path = Path("/home/adnan21/.cache/huggingface/hub/datasets--aai530-group6--ddxplus/snapshots/2ad986acc1ec62fb4a94171acc43f4fdd5bfde53/release_evidences.json")
    with open(evidence_path) as f:
        evidences_map = json.load(f)
    
    results = {"total": 0, "correct": 0, "by_specialty": {}}
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, case in enumerate(dataset):
            pathology = case["PATHOLOGY"]
            expected = get_specialty(pathology)
            
            if expected == "general_medicine" and pathology not in PATHOLOGY_TO_SPECIALTY:
                continue
            
            results["total"] += 1
            
            # Decode evidences to symptoms
            evidence_codes = extract_evidence_codes(case["EVIDENCES"])
            symptoms = []
            for code in evidence_codes[:10]:
                if code in evidences_map:
                    ev = evidences_map[code]
                    if isinstance(ev, dict):
                        q = ev.get("question_en", ev.get("name", ""))
                        q = q.replace("Do you have ", "").replace("?", "").strip()
                        if q:
                            symptoms.append(q)
            
            symptom_text = ", ".join(symptoms[:8]) if symptoms else f"symptoms of {pathology}"
            sex = "male" if case["SEX"] == "M" else "female"
            
            print(f"[{i+1}/{n_samples}] {pathology[:25]:25} -> ", end="", flush=True)
            
            try:
                response = await client.post(
                    "http://localhost:8000/api/triage",
                    json={
                        "symptoms": symptom_text,
                        "age": case["AGE"],
                        "sex": sex,
                    },
                )
                response.raise_for_status()
                data = response.json()
                
                predicted = data["primary_specialty"]
                is_correct = predicted == expected
                
                # Accept emergency for critical cases
                if expected in ["cardiology", "pulmonology"] and predicted == "emergency":
                    is_correct = True
                
                if is_correct:
                    results["correct"] += 1
                    print(f"✓ {predicted}")
                else:
                    print(f"✗ {predicted} (expected {expected})")
                
                if expected not in results["by_specialty"]:
                    results["by_specialty"][expected] = {"total": 0, "correct": 0}
                results["by_specialty"][expected]["total"] += 1
                if is_correct:
                    results["by_specialty"][expected]["correct"] += 1
                    
            except Exception as e:
                print(f"ERROR: {str(e)[:40]}")
    
    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"FULL API EVALUATION ({n_samples} samples)")
    print(f"{'='*60}")
    print(f"Accuracy: {results['correct']}/{results['total']} ({accuracy:.1%})")
    print(f"\nPer-Specialty:")
    for spec, stats in sorted(results["by_specialty"].items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {spec:20} {stats['correct']:4}/{stats['total']:4} ({acc:.0%})")
    
    return results


async def main():
    # First test classifier directly (fast, no API)
    print("=" * 60)
    print("PART 1: Classifier-Only Evaluation")
    print("=" * 60)
    await evaluate_classifier_only(n_samples=1000)
    
    # Then test full API if backend is running
    print("\n" + "=" * 60)
    print("PART 2: Full API Evaluation (requires backend running)")
    print("=" * 60)
    try:
        await evaluate_full_api(n_samples=100)
    except Exception as e:
        print(f"API test skipped (backend not running): {e}")


if __name__ == "__main__":
    asyncio.run(main())
