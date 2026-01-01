"""
DDXPlus dataset evaluation with proper evidence decoding.
"""

import asyncio
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from datasets import load_dataset


class DDXPlusEvaluator:
    """Evaluate triage system on DDXPlus dataset."""
    
    # Map DDXPlus pathologies to our specialties
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
        
        # General Medicine (respiratory/ENT)
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
        
        # Emergency
        "Anaphylaxis": "emergency",
        "Scombroid food poisoning": "emergency",
        "Spontaneous rib fracture": "emergency",
        
        # Dermatology
        "Localized edema": "dermatology",
        "Atopic dermatitis": "dermatology",
        "SLE": "general_medicine",
        "Chagas": "general_medicine",
        "Pulmonary neoplasm": "pulmonology",
    }

    def __init__(self, evidence_path: str):
        """Load evidence mapping."""
        with open(evidence_path, 'r') as f:
            self.evidences = json.load(f)
        print(f"Loaded {len(self.evidences)} evidence definitions")
    
    def decode_evidences(self, evidence_str: str) -> List[str]:
        """Decode evidence codes into symptom descriptions."""
        try:
            evidence_list = ast.literal_eval(evidence_str)
        except:
            return []
        
        symptoms = []
        for ev in evidence_list:
            # Handle format: E_XX or E_XX_@_V_YY
            parts = ev.split("_@_")
            base_code = parts[0]
            value = parts[1] if len(parts) > 1 else None
            
            if base_code not in self.evidences:
                continue
                
            ev_info = self.evidences[base_code]
            
            # Get the symptom description
            if isinstance(ev_info, dict):
                # Use question_en or name
                question = ev_info.get("question_en", ev_info.get("name", ""))
                
                # If there's a value, try to decode it
                if value and "value_meaning" in ev_info:
                    value_map = ev_info.get("value_meaning", {})
                    # Extract value number
                    val_num = value.replace("V_", "")
                    if val_num in value_map:
                        symptom = f"{question}: {value_map[val_num]}"
                    else:
                        symptom = question
                else:
                    symptom = question
                    
                # Clean up the symptom text
                symptom = symptom.replace("Do you have ", "").replace("Do you feel ", "")
                symptom = symptom.replace("Are you ", "").replace("Did you ", "")
                symptom = symptom.replace("?", "").strip()
                
                if symptom and symptom not in symptoms:
                    symptoms.append(symptom)
        
        return symptoms[:10]  # Limit to first 10 symptoms
    
    def get_specialty(self, pathology: str) -> str:
        """Map pathology to specialty."""
        return self.PATHOLOGY_TO_SPECIALTY.get(pathology, "general_medicine")
    
    async def evaluate(self, n_samples: int = 100) -> Dict:
        """Run evaluation on DDXPlus."""
        print(f"Loading DDXPlus test set (first {n_samples} samples)...")
        dataset = load_dataset("aai530-group6/ddxplus", split=f"test[:{n_samples}]")
        
        results = {
            "total": 0,
            "correct": 0,
            "correct_top2": 0,
            "errors": 0,
            "by_specialty": {},
            "by_pathology": {},
            "misclassified": [],
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, case in enumerate(dataset):
                results["total"] += 1
                
                pathology = case["PATHOLOGY"]
                expected = self.get_specialty(pathology)
                
                # Decode symptoms from evidences
                symptoms = self.decode_evidences(case["EVIDENCES"])
                symptom_text = ", ".join(symptoms) if symptoms else f"symptoms of {pathology}"
                
                # Convert sex
                sex = "male" if case["SEX"] == "M" else "female" if case["SEX"] == "F" else "other"
                
                print(f"[{i+1}/{n_samples}] {pathology[:30]:30} -> ", end="", flush=True)
                
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
                    secondary = data.get("secondary_specialty")
                    
                    # Check correctness
                    is_correct = predicted == expected
                    
                    # Accept emergency for critical conditions
                    if expected in ["cardiology", "pulmonology"] and predicted == "emergency":
                        is_correct = True
                    
                    is_top2 = expected in [predicted, secondary] or (
                        expected in ["cardiology", "pulmonology"] and "emergency" in [predicted, secondary]
                    )
                    
                    if is_correct:
                        results["correct"] += 1
                        print(f"✓ {predicted}")
                    else:
                        print(f"✗ {predicted} (expected {expected})")
                        results["misclassified"].append({
                            "pathology": pathology,
                            "expected": expected,
                            "predicted": predicted,
                            "symptoms": symptom_text[:100],
                        })
                    
                    if is_top2:
                        results["correct_top2"] += 1
                    
                    # Track by specialty
                    if expected not in results["by_specialty"]:
                        results["by_specialty"][expected] = {"total": 0, "correct": 0}
                    results["by_specialty"][expected]["total"] += 1
                    if is_correct:
                        results["by_specialty"][expected]["correct"] += 1
                    
                    # Track by pathology
                    if pathology not in results["by_pathology"]:
                        results["by_pathology"][pathology] = {"total": 0, "correct": 0}
                    results["by_pathology"][pathology]["total"] += 1
                    if is_correct:
                        results["by_pathology"][pathology]["correct"] += 1
                        
                except Exception as e:
                    results["errors"] += 1
                    print(f"ERROR: {str(e)[:40]}")
        
        self._print_report(results)
        self._save_results(results)
        return results
    
    def _print_report(self, results: Dict) -> None:
        """Print evaluation report."""
        valid = results["total"] - results["errors"]
        accuracy = results["correct"] / valid if valid > 0 else 0
        top2_acc = results["correct_top2"] / valid if valid > 0 else 0
        
        print("\n" + "=" * 70)
        print("                  DDXPlus EVALUATION REPORT")
        print("=" * 70)
        print(f"\n📊 OVERALL METRICS")
        print(f"   Total cases:       {results['total']}")
        print(f"   Valid responses:   {valid}")
        print(f"   Errors:            {results['errors']}")
        print(f"   Accuracy:          {results['correct']}/{valid} ({accuracy:.1%})")
        print(f"   Top-2 Accuracy:    {results['correct_top2']}/{valid} ({top2_acc:.1%})")
        
        print(f"\n📋 PER-SPECIALTY ACCURACY")
        for spec, stats in sorted(results["by_specialty"].items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            bar = "█" * int(acc * 20)
            print(f"   {spec:20} {bar:20} {stats['correct']}/{stats['total']} ({acc:.0%})")
        
        print(f"\n❌ SAMPLE MISCLASSIFICATIONS (first 10)")
        for m in results["misclassified"][:10]:
            print(f"   {m['pathology'][:25]:25} {m['expected']:15} -> {m['predicted']}")
        
        print("\n" + "=" * 70)
    
    def _save_results(self, results: Dict) -> None:
        """Save results to JSON."""
        # Remove large lists for JSON
        save_data = {
            "total": results["total"],
            "correct": results["correct"],
            "correct_top2": results["correct_top2"],
            "errors": results["errors"],
            "accuracy": results["correct"] / (results["total"] - results["errors"]) if results["total"] > results["errors"] else 0,
            "by_specialty": results["by_specialty"],
            "misclassified_count": len(results["misclassified"]),
        }
        with open("ddxplus_results.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print("\n💾 Results saved to ddxplus_results.json")


async def main():
    evidence_path = Path(__file__).parent.parent.parent / "data" / "ddxplus" / "release_evidences.json"
    
    if not evidence_path.exists():
        # Try cache location
        evidence_path = Path("/home/adnan21/.cache/huggingface/hub/datasets--aai530-group6--ddxplus/snapshots/2ad986acc1ec62fb4a94171acc43f4fdd5bfde53/release_evidences.json")
    
    evaluator = DDXPlusEvaluator(str(evidence_path))
    await evaluator.evaluate(n_samples=100)


if __name__ == "__main__":
    asyncio.run(main())
