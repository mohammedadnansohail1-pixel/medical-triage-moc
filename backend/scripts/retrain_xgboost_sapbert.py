"""
Retrain XGBoost on SapBERT-recovered evidence codes.
Fixes distribution mismatch: trained on original codes, inference uses SapBERT codes.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.sapbert_linker import get_sapbert_linker


def load_data():
    with open("data/classifier/train_data.pkl", "rb") as f:
        train = pickle.load(f)
    with open("data/classifier/test_data.pkl", "rb") as f:
        test = pickle.load(f)
    with open("data/classifier/vocabulary.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("/home/adnan21/projects/medical-triage-moc/data/ddxplus/release_evidences.json") as f:
        evidences = json.load(f)
    return train, test, vocab, evidences


def convert_codes_to_text(sample_x, idx_to_code, evidences, max_symptoms=20):
    active_codes = [idx_to_code[j] for j in range(len(sample_x)) if sample_x[j] == 1 and j in idx_to_code]
    return [evidences[code].get("question_en", "") for code in active_codes[:max_symptoms] if code in evidences]


def sapbert_vectorize(symptoms, sapbert, code_to_idx, num_features=225, threshold=0.3):
    if not symptoms:
        return np.zeros(num_features, dtype=np.float32)
    
    matches = sapbert.link_symptoms(symptoms, top_k=3, threshold=threshold)
    features = np.zeros(num_features, dtype=np.float32)
    
    for symptom, code, score in matches:
        if code in code_to_idx:
            features[code_to_idx[code]] = 1.0
    
    return features


def main():
    print("=" * 60)
    print("RETRAINING XGBOOST ON SAPBERT-RECOVERED CODES")
    print("=" * 60)
    
    print("\n[1/5] Loading data...")
    train, test, vocab, evidences = load_data()
    
    idx_to_code = {v: k for k, v in vocab["code_to_idx"].items()}
    idx_to_specialty = vocab["idx_to_specialty"]
    code_to_idx = vocab["code_to_idx"]
    
    print(f"  Train samples: {len(train['X'])}")
    print(f"  Test samples: {len(test['X'])}")
    
    print("\n[2/5] Loading SapBERT...")
    sapbert = get_sapbert_linker()
    sapbert.load()
    sapbert.build_evidence_index(Path("/home/adnan21/projects/medical-triage-moc/data/ddxplus/release_evidences.json"))
    
    print("\n[3/5] Converting training data through SapBERT...")
    
    n_train = 10000
    indices = np.random.choice(len(train['X']), n_train, replace=False)
    
    X_sapbert = []
    y_sapbert = []
    
    for i, idx in enumerate(indices):
        if i % 1000 == 0:
            print(f"  Processing {i}/{n_train}...")
        
        symptoms = convert_codes_to_text(train['X'][idx], idx_to_code, evidences, max_symptoms=20)
        
        if not symptoms:
            continue
        
        features = sapbert_vectorize(symptoms, sapbert, code_to_idx)
        X_sapbert.append(features)
        y_sapbert.append(train['y'][idx])
    
    X_sapbert = np.array(X_sapbert)
    y_sapbert = np.array(y_sapbert)
    
    print(f"  Generated {len(X_sapbert)} samples")
    
    print("\n[4/5] Training XGBoost...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_sapbert, y_sapbert, test_size=0.1, random_state=42, stratify=y_sapbert
    )
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(idx_to_specialty),
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    
    print("\n[5/5] Evaluating on test set...")
    
    n_test = 500
    X_test_sapbert = []
    y_test = []
    
    for i in range(n_test):
        symptoms = convert_codes_to_text(test['X'][i], idx_to_code, evidences, max_symptoms=20)
        if not symptoms:
            continue
        features = sapbert_vectorize(symptoms, sapbert, code_to_idx)
        X_test_sapbert.append(features)
        y_test.append(test['y'][i])
    
    X_test_sapbert = np.array(X_test_sapbert)
    y_test = np.array(y_test)
    
    y_pred = model.predict(X_test_sapbert)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.1%}")
    
    print("\nPer-specialty breakdown:")
    print(classification_report(
        y_test, y_pred,
        target_names=[idx_to_specialty[i] for i in range(len(idx_to_specialty))],
    ))
    
    output_path = Path("data/classifier/model_sapbert_v2.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {output_path}")
    
    sapbert.unload()


if __name__ == "__main__":
    main()
