"""
Train XGBoost classifier on DDXPlus data.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


def train_classifier(
    data_dir: str = "data/classifier",
    output_path: str = "data/classifier/model.pkl"
) -> float:
    """
    Train XGBoost classifier on prepared DDXPlus data.
    """
    data_path = Path(data_dir)
    
    print("Loading training data...")
    with open(data_path / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    with open(data_path / "test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    with open(data_path / "vocabulary.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]
    idx_to_specialty = vocab["idx_to_specialty"]
    n_classes = len(idx_to_specialty)
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {n_classes}")
    
    # Train XGBoost
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {accuracy:.1%}")
    print(f"{'='*60}")
    
    # Per-class report
    target_names = [idx_to_specialty[i] for i in range(n_classes)]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
    
    # Save model
    print(f"\nSaving model to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Training complete!")
    return accuracy


if __name__ == "__main__":
    train_classifier()
