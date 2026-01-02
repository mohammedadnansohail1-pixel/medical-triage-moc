#!/usr/bin/env python3
"""Retrain XGBoost on SapBERT-Recovered Codes."""

import logging
import pickle
import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Config:
    n_train_samples: int = 50000
    n_test_samples: int = 5000
    val_split: float = 0.15
    random_seed: int = 42
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.1
    calibration_method: str = "isotonic"
    output_dir: Path = Path("backend/data/classifier")
    model_filename: str = "model_sapbert_aligned.pkl"
    sapbert_noise_rate: float = 0.15

PATHOLOGY_TO_SPECIALTY: Dict[str, str] = {
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
SPECIALTY_TO_IDX = {s: i for i, s in enumerate(SPECIALTIES)}
IDX_TO_SPECIALTY = {i: s for s, i in SPECIALTY_TO_IDX.items()}

def load_ddxplus(n_train: int, n_test: int):
    logger.info(f"Loading DDXPlus (train={n_train}, test={n_test})...")
    train_ds = load_dataset("aai530-group6/ddxplus", split=f"train[:{n_train}]")
    test_ds = load_dataset("aai530-group6/ddxplus", split=f"test[:{n_test}]")
    logger.info(f"Loaded {len(train_ds)} train, {len(test_ds)} test")
    return train_ds, test_ds

def extract_codes(evidence_str: str) -> List[str]:
    try:
        evidences = ast.literal_eval(evidence_str)
        return [ev.split("_@_")[0] for ev in evidences]
    except:
        return []

def build_vocabulary(datasets) -> Tuple[Dict[str, int], List[str]]:
    all_codes = set()
    for ds in datasets:
        for row in ds:
            codes = extract_codes(row["EVIDENCES"])
            all_codes.update(codes)
    all_codes = sorted(all_codes)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    logger.info(f"Vocabulary: {len(all_codes)} codes")
    return code_to_idx, all_codes

def simulate_sapbert_noise(codes: List[str], all_codes: List[str], noise_rate: float, rng) -> List[str]:
    noisy = []
    for code in codes:
        r = rng.random()
        if r < noise_rate * 0.5:
            continue
        elif r < noise_rate:
            noisy.append(rng.choice(all_codes))
        else:
            noisy.append(code)
    if rng.random() < noise_rate * 0.3 and all_codes:
        noisy.append(rng.choice(all_codes))
    return noisy

def codes_to_vector(codes: List[str], code_to_idx: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(code_to_idx), dtype=np.float32)
    for code in codes:
        if code in code_to_idx:
            vec[code_to_idx[code]] = 1.0
    return vec

def prepare_features(dataset, code_to_idx, all_codes, add_noise, noise_rate, seed):
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    skipped = 0
    for row in dataset:
        pathology = row["PATHOLOGY"]
        specialty = PATHOLOGY_TO_SPECIALTY.get(pathology)
        if specialty is None:
            skipped += 1
            continue
        codes = extract_codes(row["EVIDENCES"])
        if add_noise:
            codes = simulate_sapbert_noise(codes, all_codes, noise_rate, rng)
        X_list.append(codes_to_vector(codes, code_to_idx))
        y_list.append(SPECIALTY_TO_IDX[specialty])
    if skipped:
        logger.warning(f"Skipped {skipped} unknown pathologies")
    return np.array(X_list), np.array(y_list)

def compute_class_weights(y):
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    return {cls: total / (n_classes * count) for cls, count in counts.items()}

def train_xgboost(X_train, y_train, X_val, y_val, config):
    logger.info("Training XGBoost...")
    weights = compute_class_weights(y_train)
    sample_weights = np.array([weights[y] for y in y_train])
    model = xgb.XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        objective="multi:softprob",
        num_class=len(SPECIALTIES),
        eval_metric="mlogloss",
        random_state=config.random_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=True)
    y_pred = model.predict(X_val)
    logger.info(f"Val Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    logger.info(f"Val Macro F1: {f1_score(y_val, y_pred, average='macro'):.4f}")
    return model

def calibrate(model, X_cal, y_cal, method):
    """Apply calibration using cross-validation approach."""
    logger.info(f"Calibrating with {method}...")
    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=3)
    calibrated.fit(X_cal, y_cal)
    return calibrated

def compute_ece(y_true, y_proba, n_bins=10):
    confidences = np.max(y_proba, axis=1)
    accuracies = (np.argmax(y_proba, axis=1) == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            ece += np.abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.mean()
    return ece

def evaluate(model, X, y, name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    ece = compute_ece(y, y_proba)
    logger.info(f"\n{name}: Accuracy={acc:.4f}, Macro F1={macro_f1:.4f}, ECE={ece:.4f}")
    logger.info(f"\n{classification_report(y, y_pred, target_names=SPECIALTIES)}")
    return {"accuracy": acc, "macro_f1": macro_f1, "ece": ece}

def main():
    config = Config()
    logger.info("=" * 60)
    logger.info("RETRAINING XGBOOST ON SAPBERT-RECOVERED CODES")
    logger.info("=" * 60)
    
    train_ds, test_ds = load_ddxplus(config.n_train_samples, config.n_test_samples)
    code_to_idx, all_codes = build_vocabulary([train_ds, test_ds])
    
    X_train_noisy, y_train = prepare_features(train_ds, code_to_idx, all_codes, True, config.sapbert_noise_rate, config.random_seed)
    X_test_noisy, y_test = prepare_features(test_ds, code_to_idx, all_codes, True, config.sapbert_noise_rate, config.random_seed + 1)
    X_test_clean, _ = prepare_features(test_ds, code_to_idx, all_codes, False, 0, config.random_seed)
    
    X_train, X_val, y_train_split, y_val = train_test_split(X_train_noisy, y_train, test_size=config.val_split, random_state=config.random_seed, stratify=y_train)
    
    model = train_xgboost(X_train, y_train_split, X_val, y_val, config)
    metrics_before = evaluate(model, X_test_noisy, y_test, "Before Calibration")
    
    # Calibrate on validation set
    calibrated = calibrate(model, X_val, y_val, config.calibration_method)
    metrics_after = evaluate(calibrated, X_test_noisy, y_test, "After Calibration")
    metrics_clean = evaluate(calibrated, X_test_clean, y_test, "On Clean Data")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.output_dir / config.model_filename
    with open(model_path, "wb") as f:
        pickle.dump({"model": calibrated, "code_to_idx": code_to_idx, "all_codes": all_codes,
                     "specialty_to_idx": SPECIALTY_TO_IDX, "idx_to_specialty": IDX_TO_SPECIALTY,
                     "metrics": {"before": metrics_before, "after": metrics_after, "clean": metrics_clean}}, f)
    
    logger.info(f"\nModel saved to {model_path}")
    logger.info(f"\nFINAL RESULTS:")
    logger.info(f"  Before Calibration: {metrics_before['accuracy']:.1%} accuracy, {metrics_before['macro_f1']:.1%} F1")
    logger.info(f"  After Calibration:  {metrics_after['accuracy']:.1%} accuracy, {metrics_after['macro_f1']:.1%} F1")
    logger.info(f"  On Clean Data:      {metrics_clean['accuracy']:.1%} accuracy, {metrics_clean['macro_f1']:.1%} F1")

if __name__ == "__main__":
    main()
