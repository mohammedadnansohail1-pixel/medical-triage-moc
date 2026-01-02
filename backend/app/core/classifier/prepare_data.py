"""
Prepare DDXPlus data for classifier training.
Converts evidences to symptom vectors and maps pathologies to specialties.
"""

import ast
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset


# Pathology to specialty mapping
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

# Only specialties present in DDXPlus (no orthopedics)
SPECIALTIES = [
    "cardiology",
    "dermatology",
    "emergency",
    "gastroenterology",
    "general_medicine",
    "neurology",
    "pulmonology",
]


def extract_evidence_codes(evidence_str: str) -> List[str]:
    """Extract base evidence codes from evidence string."""
    try:
        evidences = ast.literal_eval(evidence_str)
        codes = []
        for ev in evidences:
            base = ev.split("_@_")[0]
            if base not in codes:
                codes.append(base)
        return codes
    except:
        return []


def prepare_training_data(
    n_train: int = 50000,
    n_test: int = 5000,
    output_dir: str = "data/classifier"
) -> Tuple[str, str]:
    """
    Prepare training and test data from DDXPlus.
    """
    print(f"Loading DDXPlus dataset...")
    train_ds = load_dataset("aai530-group6/ddxplus", split=f"train[:{n_train}]")
    test_ds = load_dataset("aai530-group6/ddxplus", split=f"test[:{n_test}]")
    
    # Build vocabulary of all evidence codes
    print("Building evidence vocabulary...")
    all_codes = set()
    for ds in [train_ds, test_ds]:
        for sample in ds:
            codes = extract_evidence_codes(sample["EVIDENCES"])
            all_codes.update(codes)
    
    code_to_idx = {code: idx for idx, code in enumerate(sorted(all_codes))}
    specialty_to_idx = {spec: idx for idx, spec in enumerate(SPECIALTIES)}
    
    print(f"  Evidence codes: {len(code_to_idx)}")
    print(f"  Specialties: {len(specialty_to_idx)}")
    print(f"  Specialty mapping: {specialty_to_idx}")
    
    def process_dataset(ds, name: str):
        """Process dataset into feature vectors and labels."""
        X = []
        y = []
        skipped = 0
        
        for sample in ds:
            pathology = sample["PATHOLOGY"]
            specialty = PATHOLOGY_TO_SPECIALTY.get(pathology)
            
            if specialty is None or specialty not in specialty_to_idx:
                skipped += 1
                continue
            
            # Create feature vector
            codes = extract_evidence_codes(sample["EVIDENCES"])
            feature_vec = np.zeros(len(code_to_idx), dtype=np.float32)
            for code in codes:
                if code in code_to_idx:
                    feature_vec[code_to_idx[code]] = 1.0
            
            # Add age and sex
            age = sample["AGE"] / 100.0
            sex = 1.0 if sample["SEX"] == "M" else 0.0
            
            X.append(np.append(feature_vec, [age, sex]))
            y.append(specialty_to_idx[specialty])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"  {name}: {len(X)} samples, {skipped} skipped")
        print(f"  Label distribution: {np.bincount(y)}")
        return X, y
    
    print("Processing training data...")
    X_train, y_train = process_dataset(train_ds, "train")
    
    print("Processing test data...")
    X_test, y_test = process_dataset(test_ds, "test")
    
    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / "train_data.pkl"
    test_path = output_path / "test_data.pkl"
    vocab_path = output_path / "vocabulary.pkl"
    
    with open(train_path, "wb") as f:
        pickle.dump({"X": X_train, "y": y_train}, f)
    
    with open(test_path, "wb") as f:
        pickle.dump({"X": X_test, "y": y_test}, f)
    
    with open(vocab_path, "wb") as f:
        pickle.dump({
            "code_to_idx": code_to_idx,
            "specialty_to_idx": specialty_to_idx,
            "idx_to_specialty": {v: k for k, v in specialty_to_idx.items()},
        }, f)
    
    print(f"\n✅ Data saved to {output_dir}/")
    return str(train_path), str(test_path)


if __name__ == "__main__":
    prepare_training_data()
