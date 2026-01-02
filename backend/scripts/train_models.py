"""
Train all models from scratch using DDXPlus data.

This script generates:
- backend/data/classifier/vocabulary.pkl
- backend/data/classifier/model.pkl  
- backend/data/classifier/train_data.pkl
- backend/data/classifier/test_data.pkl
"""
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DDXPLUS_DIR = PROJECT_ROOT / "data" / "ddxplus"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "classifier"


# Pathology to specialty mapping
PATHOLOGY_TO_SPECIALTY = {
    # Emergency
    "Pulmonary embolism": "emergency",
    "Myocardial infarction": "emergency",
    "Anaphylaxis": "emergency",
    "Acute pulmonary edema": "emergency",
    "Guillain-Barré syndrome": "emergency",
    "Boerhaave": "emergency",
    "Chagas": "emergency",
    "SLE": "emergency",
    
    # Cardiology
    "Unstable angina": "cardiology",
    "Stable angina": "cardiology",
    "Atrial fibrillation": "cardiology",
    "Myocarditis": "cardiology",
    "PSVT": "cardiology",
    "Possible NSTEMI / STEMI": "cardiology",
    "Pericarditis": "cardiology",
    "Sarcoidosis": "cardiology",
    
    # Pulmonology
    "Pneumonia": "pulmonology",
    "Bronchitis": "pulmonology",
    "COPD": "pulmonology",
    "Acute COPD exacerbation / infection": "pulmonology",
    "Tuberculosis": "pulmonology",
    "Bronchiectasis": "pulmonology",
    "Croup": "pulmonology",
    "Bronchospasm / acute asthma exacerbation": "pulmonology",
    "Viral pharyngitis": "pulmonology",
    "Whooping cough": "pulmonology",
    "Acute laryngitis": "pulmonology",
    "Influenza": "pulmonology",
    "Spontaneous pneumothorax": "pulmonology",
    "Pulmonary neoplasm": "pulmonology",
    "Allergic sinusitis": "pulmonology",
    "Chronic rhinosinusitis": "pulmonology",
    "Acute rhinosinusitis": "pulmonology",
    
    # Neurology
    "Cluster headache": "neurology",
    "Spontaneous rib fracture": "neurology",
    "Panic attack": "neurology",
    "Anemia": "neurology",
    "Myasthenia gravis": "neurology",
    
    # Gastroenterology
    "GERD": "gastroenterology",
    "Pancreatic neoplasm": "gastroenterology",
    "HIV (initial infection)": "gastroenterology",
    "Acute otitis media": "gastroenterology",
    "Epiglottitis": "gastroenterology",
    "Inguinal hernia": "gastroenterology",
    "Scombroid food poisoning": "gastroenterology",
    
    # Dermatology
    "Localized edema": "dermatology",
    "Atopic dermatitis": "dermatology",
    
    # General Medicine (catch-all)
    "URTI": "general_medicine",
    "Larygospasm": "general_medicine",
    "Acute dystonic reactions": "general_medicine",
    "Ebola": "general_medicine",
    "Bronchiolitis": "general_medicine",
}

# Default specialty for unmapped pathologies
DEFAULT_SPECIALTY = "general_medicine"

# All specialties
SPECIALTIES = [
    "cardiology",
    "dermatology",
    "emergency",
    "gastroenterology",
    "general_medicine",
    "neurology",
    "pulmonology",
]


def load_ddxplus_data():
    """Load DDXPlus evidences and conditions."""
    print("Loading DDXPlus data...")
    
    evidences_path = DDXPLUS_DIR / "release_evidences.json"
    conditions_path = DDXPLUS_DIR / "release_conditions.json"
    probs_path = DDXPLUS_DIR / "symptom_condition_probs.json"
    
    if not evidences_path.exists():
        raise FileNotFoundError(f"DDXPlus evidences not found at {evidences_path}")
    if not conditions_path.exists():
        raise FileNotFoundError(f"DDXPlus conditions not found at {conditions_path}")
    
    with open(evidences_path) as f:
        evidences = json.load(f)
    
    with open(conditions_path) as f:
        conditions = json.load(f)
    
    # Load probability matrix
    symptom_probs = None
    if probs_path.exists():
        with open(probs_path) as f:
            symptom_probs = json.load(f)
        print(f"  Loaded symptom probabilities for {len(symptom_probs['conditions'])} conditions")
    
    print(f"  Loaded {len(evidences)} evidence codes")
    print(f"  Loaded {len(conditions)} conditions")
    
    return evidences, conditions, symptom_probs


def generate_training_data(evidences, conditions, symptom_probs, n_samples_per_condition=1000):
    """Generate training samples from DDXPlus data."""
    print(f"\nGenerating training samples ({n_samples_per_condition} per condition)...")
    
    # Build vocabulary - sort for consistency
    all_codes = sorted(evidences.keys())
    code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
    n_features = len(code_to_idx) + 2  # +2 for age, sex
    
    specialty_to_idx = {spec: idx for idx, spec in enumerate(SPECIALTIES)}
    idx_to_specialty = {idx: spec for spec, idx in specialty_to_idx.items()}
    
    # Get condition-to-specialty mapping
    condition_specialty = {}
    for cond_key, cond_data in conditions.items():
        name = cond_data.get("cond_name_eng", cond_key)
        specialty = PATHOLOGY_TO_SPECIALTY.get(name, DEFAULT_SPECIALTY)
        condition_specialty[name] = specialty
    
    # Use probability matrix from symptom_probs
    prob_matrix = symptom_probs["probabilities"]
    
    X = []
    y = []
    
    np.random.seed(42)
    
    for condition_name, specialty in condition_specialty.items():
        if condition_name not in prob_matrix:
            print(f"  Warning: {condition_name} not in probability matrix")
            continue
            
        specialty_idx = specialty_to_idx[specialty]
        condition_probs = prob_matrix[condition_name]
        
        # Find symptoms with non-zero probability
        active_symptoms = [(code, prob) for code, prob in condition_probs.items() if prob > 0]
        
        for _ in range(n_samples_per_condition):
            # Create feature vector
            features = np.zeros(n_features, dtype=np.float32)
            
            # Random age (normalized to 0-1)
            features[-2] = np.random.uniform(0.2, 0.9)
            
            # Random sex (0 or 1)
            features[-1] = np.random.choice([0, 1])
            
            # Sample symptoms based on probabilities
            for code, prob in active_symptoms:
                if code in code_to_idx:
                    # Use probability to determine if symptom is present
                    # Add some noise to make training more robust
                    adjusted_prob = min(1.0, prob * np.random.uniform(0.8, 1.2))
                    if np.random.random() < adjusted_prob:
                        features[code_to_idx[code]] = 1.0
            
            # Ensure at least some symptoms
            if features[:-2].sum() < 2:
                # Add random symptoms from the condition's active set
                for code, _ in np.random.permutation(active_symptoms)[:3]:
                    if code in code_to_idx:
                        features[code_to_idx[code]] = 1.0
            
            X.append(features)
            y.append(specialty_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"  Generated {len(X)} samples")
    print(f"  Feature dimension: {X.shape[1]}")
    
    # Class distribution
    print("  Class distribution:")
    for spec, idx in specialty_to_idx.items():
        count = (y == idx).sum()
        print(f"    {spec}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, code_to_idx, idx_to_specialty


def train_classifier(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier."""
    print("\nTraining XGBoost classifier...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(SPECIALTIES),
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Test accuracy: {test_acc:.1%}")
    
    return model


def save_artifacts(model, X_train, y_train, X_test, y_test, code_to_idx, idx_to_specialty):
    """Save all artifacts."""
    print("\nSaving artifacts...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Vocabulary
    vocab = {
        "code_to_idx": code_to_idx,
        "idx_to_specialty": idx_to_specialty,
    }
    with open(OUTPUT_DIR / "vocabulary.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"  Saved vocabulary.pkl")
    
    # Model
    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model.pkl")
    
    # Training data
    with open(OUTPUT_DIR / "train_data.pkl", "wb") as f:
        pickle.dump({"X": X_train, "y": y_train}, f)
    print(f"  Saved train_data.pkl")
    
    # Test data
    with open(OUTPUT_DIR / "test_data.pkl", "wb") as f:
        pickle.dump({"X": X_test, "y": y_test}, f)
    print(f"  Saved test_data.pkl")


def main():
    print("=" * 60)
    print("Medical Triage AI - Model Training")
    print("=" * 60)
    
    # Load data
    evidences, conditions, symptom_probs = load_ddxplus_data()
    
    if symptom_probs is None:
        raise RuntimeError("symptom_condition_probs.json required for training")
    
    # Generate training data
    X, y, code_to_idx, idx_to_specialty = generate_training_data(
        evidences, conditions, symptom_probs, n_samples_per_condition=1000
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"\nSplit: {len(X_train)} train, {len(X_test)} test")
    
    # Train model
    model = train_classifier(X_train, y_train, X_test, y_test)
    
    # Save artifacts
    save_artifacts(model, X_train, y_train, X_test, y_test, code_to_idx, idx_to_specialty)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
