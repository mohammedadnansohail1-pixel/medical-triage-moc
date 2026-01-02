"""
Evaluate differential diagnosis accuracy on DDXPlus.

Metrics:
- Top-1: Exact condition match
- Top-3: Correct condition in top 3
- Top-5: Correct condition in top 5
- Per-specialty breakdown
"""

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

from app.core.specialty_agent import get_specialty_manager
from app.core.classifier.router import ClassifierRouter


PATHOLOGY_TO_SPECIALTY = {
    "Possible NSTEMI / STEMI": "cardiology",
    "Unstable angina": "cardiology",
    "Stable angina": "cardiology",
    "Myocarditis": "cardiology",
    "Pericarditis": "cardiology",
    "Atrial fibrillation": "cardiology",
    "SVT": "cardiology",
    "PSVT": "cardiology",
    "Bronchitis": "pulmonology",
    "Pneumonia": "pulmonology",
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
    "Bronchiolitis": "pulmonology",
    "Pulmonary embolism": "pulmonology",
    "URTI": "pulmonology",
    "Influenza": "pulmonology",
    "GERD": "gastroenterology",
    "Boerhaave": "gastroenterology",
    "Pancreatic neoplasm": "gastroenterology",
    "Inguinal hernia": "gastroenterology",
    "Cluster headache": "neurology",
    "Panic attack": "neurology",
    "Guillain-Barré syndrome": "neurology",
    "Myasthenia gravis": "neurology",
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
    "Localized edema": "general_medicine",
    "Anaphylaxis": "emergency",
    "Scombroid food poisoning": "emergency",
    "Spontaneous rib fracture": "emergency",
    "Ebola": "emergency",
    "Atopic dermatitis": "dermatology",
}


def get_specialty(pathology: str) -> str:
    return PATHOLOGY_TO_SPECIALTY.get(pathology, "general_medicine")


def extract_evidence_codes(row) -> List[str]:
    """Extract base evidence codes from patient."""
    try:
        evidences = ast.literal_eval(row['EVIDENCES'])
        return list(set(ev.split("_@_")[0] for ev in evidences))
    except:
        return []


@dataclass
class EvalResults:
    n_samples: int
    # Overall metrics (given correct specialty)
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    # Specialty routing accuracy
    specialty_accuracy: float
    # Per-specialty DDx accuracy
    per_specialty: Dict[str, Dict[str, float]]


def evaluate(n_samples: int = 500, verbose: bool = True) -> EvalResults:
    """
    Evaluate differential diagnosis on DDXPlus test set.
    
    Two-tier evaluation:
    1. Given CORRECT specialty, how accurate is DDx?
    2. Given PREDICTED specialty, end-to-end accuracy
    """
    if verbose:
        print(f"Loading DDXPlus test set ({n_samples} samples)...")
    
    dataset = load_dataset("aai530-group6/ddxplus", split=f"test[:{n_samples}]")
    
    # Load components
    model_path = Path("/home/adnan21/projects/medical-triage-moc/data/ddxplus/condition_model.json")
    manager = get_specialty_manager(model_path)
    router = ClassifierRouter()
    
    # Counters
    specialty_correct = 0
    
    # Given correct specialty
    top1_given_correct = 0
    top3_given_correct = 0
    top5_given_correct = 0
    n_with_correct_specialty = 0
    
    # Per-specialty
    per_specialty_top1 = defaultdict(int)
    per_specialty_top3 = defaultdict(int)
    per_specialty_total = defaultdict(int)
    
    for i, row in enumerate(dataset):
        if verbose and i % 100 == 0:
            print(f"  Processing {i}/{n_samples}...")
        
        true_pathology = row['PATHOLOGY']
        true_specialty = get_specialty(true_pathology)
        evidence_codes = extract_evidence_codes(row)
        
        if not evidence_codes:
            continue
        
        age = row.get('AGE', 40)
        sex = 'male' if row.get('SEX') == 'M' else 'female'
        
        # Tier 1: Specialty routing
        routing = router.route_from_evidences(evidence_codes, age=age)
        pred_specialty = routing.primary_specialty
        
        if pred_specialty == true_specialty:
            specialty_correct += 1
        
        # Tier 2: DDx within TRUE specialty (oracle)
        if true_specialty in manager.available_specialties:
            n_with_correct_specialty += 1
            per_specialty_total[true_specialty] += 1
            
            ddx = manager.diagnose(
                specialty=true_specialty,
                symptom_codes=evidence_codes,
                age=age,
                sex=sex,
                top_k=5
            )
            
            predicted_conditions = [c for c, _ in ddx]
            
            if predicted_conditions and predicted_conditions[0] == true_pathology:
                top1_given_correct += 1
                per_specialty_top1[true_specialty] += 1
            
            if true_pathology in predicted_conditions[:3]:
                top3_given_correct += 1
                per_specialty_top3[true_specialty] += 1
            
            if true_pathology in predicted_conditions[:5]:
                top5_given_correct += 1
    
    # Calculate metrics
    total = len(dataset)
    
    results = EvalResults(
        n_samples=total,
        specialty_accuracy=specialty_correct / total if total > 0 else 0,
        top1_accuracy=top1_given_correct / n_with_correct_specialty if n_with_correct_specialty > 0 else 0,
        top3_accuracy=top3_given_correct / n_with_correct_specialty if n_with_correct_specialty > 0 else 0,
        top5_accuracy=top5_given_correct / n_with_correct_specialty if n_with_correct_specialty > 0 else 0,
        per_specialty={
            spec: {
                "top1": per_specialty_top1[spec] / per_specialty_total[spec] if per_specialty_total[spec] > 0 else 0,
                "top3": per_specialty_top3[spec] / per_specialty_total[spec] if per_specialty_total[spec] > 0 else 0,
                "n": per_specialty_total[spec],
            }
            for spec in per_specialty_total
        }
    )
    
    return results


def print_results(results: EvalResults):
    print("\n" + "=" * 60)
    print("DIFFERENTIAL DIAGNOSIS EVALUATION")
    print("=" * 60)
    
    print(f"\nSamples: {results.n_samples}")
    print(f"\nTier 1 - Specialty Routing: {results.specialty_accuracy:.1%}")
    
    print(f"\nTier 2 - DDx Accuracy (given correct specialty):")
    print(f"  Top-1: {results.top1_accuracy:.1%}")
    print(f"  Top-3: {results.top3_accuracy:.1%}")
    print(f"  Top-5: {results.top5_accuracy:.1%}")
    
    print(f"\nPer-Specialty DDx (Top-1 / Top-3):")
    for spec, data in sorted(results.per_specialty.items()):
        print(f"  {spec:<20}: {data['top1']:5.1%} / {data['top3']:5.1%}  (n={data['n']})")


if __name__ == "__main__":
    results = evaluate(n_samples=500, verbose=True)
    print_results(results)
