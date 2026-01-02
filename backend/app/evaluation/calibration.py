"""Calibration Metrics for Medical Triage System."""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np


@dataclass
class CalibrationResult:
    """Calibration analysis results."""
    brier_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    bin_confidences: List[float]
    bin_accuracies: List[float]
    bin_counts: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brier_score": self.brier_score,
            "ece": self.ece,
            "mce": self.mce,
        }


def compute_calibration(
    y_true: List[str],
    y_pred: List[str],
    confidences: List[float],
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute calibration metrics."""
    n = len(y_true)
    correct = np.array([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    conf = np.array(confidences)
    
    # Brier score (simplified for top-1)
    brier = float(np.mean((conf - correct) ** 2))
    
    # Binned calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
        
        count = int(mask.sum())
        bin_counts.append(count)
        
        if count > 0:
            bin_conf = float(conf[mask].mean())
            bin_acc = float(correct[mask].mean())
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
            
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)
    
    return CalibrationResult(
        brier_score=brier,
        ece=float(ece),
        mce=float(mce),
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
    )
