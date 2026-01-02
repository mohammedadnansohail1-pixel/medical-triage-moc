#!/usr/bin/env python3
"""End-to-end test on larger DDXPlus test set."""

import ast
import sys
sys.path.insert(0, "backend")

from collections import defaultdict
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from app.core.classifier_v2 import get_classifier

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
    "Anemia": "general_medicine",
    "HIV (initial infection)": "general_medicine",
    "Sarcoidosis": "general_medicine",
    "Acute dystonic reactions": "general_medicine",
    "SLE": "general_medicine",
    "Chagas": "general_medicine",
    "Anaphylaxis": "emergency",
    "Scombroid food poisoning": "emergency",
    "Spontaneous rib fracture": "emergency",
    "Ebola": "emergency",
    "Localized edema": "dermatology",
}

SPECIALTIES = ["cardiology", "dermatology", "emergency", "gastroenterology",
               "general_medicine", "neurology", "pulmonology"]

def extract_codes(evidence_str):
    try:
        evidences = ast.literal_eval(evidence_str)
        return [ev.split("_@_")[0] for ev in evidences]
    except:
        return []

def main():
    N_SAMPLES = 10000
    
    print("=" * 60)
    print(f"END-TO-END TEST ON {N_SAMPLES} SAMPLES")
    print("=" * 60)
    
    print(f"\nLoading DDXPlus test set ({N_SAMPLES} samples)...")
    test_ds = load_dataset("aai530-group6/ddxplus", split=f"test[:{N_SAMPLES}]")
    print(f"Loaded {len(test_ds)} samples")
    
    clf = get_classifier()
    print(f"Classifier loaded: {clf.get_model_info()['vocabulary_size']} codes")
    
    y_true = []
    y_pred = []
    skipped = 0
    per_specialty = defaultdict(lambda: {"correct": 0, "total": 0})
    errors = []
    
    print("\nRunning predictions...")
    for i, row in enumerate(test_ds):
        pathology = row["PATHOLOGY"]
        true_specialty = PATHOLOGY_TO_SPECIALTY.get(pathology)
        
        if true_specialty is None:
            skipped += 1
            continue
        
        codes = extract_codes(row["EVIDENCES"])
        result = clf.predict(codes)
        pred_specialty = result["specialty"]
        
        y_true.append(true_specialty)
        y_pred.append(pred_specialty)
        
        per_specialty[true_specialty]["total"] += 1
        if pred_specialty == true_specialty:
            per_specialty[true_specialty]["correct"] += 1
        else:
            if len(errors) < 10:
                errors.append({
                    "pathology": pathology,
                    "true": true_specialty,
                    "pred": pred_specialty,
                    "confidence": result["confidence"],
                    "codes": codes[:5]
                })
        
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(test_ds)}...")
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=SPECIALTIES)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=SPECIALTIES)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall ({len(y_true)} samples):")
    print(f"  Accuracy:    {accuracy:.2%}")
    print(f"  Macro F1:    {macro_f1:.2%}")
    print(f"  Weighted F1: {weighted_f1:.2%}")
    print(f"  Errors:      {len(y_true) - sum(1 for t,p in zip(y_true,y_pred) if t==p)}")
    print(f"  Skipped:     {skipped}")
    
    print(f"\nPer-Specialty:")
    for spec in SPECIALTIES:
        data = per_specialty[spec]
        if data["total"] > 0:
            acc = data["correct"] / data["total"]
            err = data["total"] - data["correct"]
            print(f"  {spec:<20}: {data['correct']:4d}/{data['total']:4d} ({acc:.2%}) - {err} errors")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=SPECIALTIES, target_names=SPECIALTIES, digits=4))
    
    if errors:
        print(f"\nSample Errors (first {len(errors)}):")
        for e in errors:
            print(f"  {e['pathology']}: {e['true']} -> {e['pred']} ({e['confidence']:.1%})")

if __name__ == "__main__":
    main()
