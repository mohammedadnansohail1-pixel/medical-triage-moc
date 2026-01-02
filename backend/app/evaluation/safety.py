"""Safety Metrics for Medical Triage System."""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class SafetyMetrics:
    """Safety-critical metrics."""
    emergency_sensitivity: float
    emergency_specificity: float
    under_triage_rate: float
    over_triage_rate: float
    triage_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emergency_sensitivity": self.emergency_sensitivity,
            "under_triage_rate": self.under_triage_rate,
            "over_triage_rate": self.over_triage_rate,
        }

    def passes_safety_threshold(
        self,
        min_emergency_sens: float = 0.95,
        max_under_triage: float = 0.05,
    ) -> Dict[str, bool]:
        return {
            "emergency_sensitivity": self.emergency_sensitivity >= min_emergency_sens,
            "under_triage_rate": self.under_triage_rate <= max_under_triage,
            "overall_safe": (
                self.emergency_sensitivity >= min_emergency_sens and
                self.under_triage_rate <= max_under_triage
            ),
        }


URGENCY = {
    "emergency": 0,
    "cardiology": 1,
    "pulmonology": 1,
    "neurology": 1,
    "gastroenterology": 2,
    "general_medicine": 2,
}


def compute_safety_metrics(
    y_true: List[str],
    y_pred: List[str],
    emergency_label: str = "emergency",
) -> SafetyMetrics:
    """Compute safety-critical metrics."""
    n = len(y_true)
    
    # Emergency detection
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == emergency_label and p == emergency_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == emergency_label and p != emergency_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != emergency_label and p == emergency_label)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != emergency_label and p != emergency_label)
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    
    # Triage errors
    under = over = correct = 0
    for t, p in zip(y_true, y_pred):
        t_urg = URGENCY.get(t, 2)
        p_urg = URGENCY.get(p, 2)
        if p_urg > t_urg:
            under += 1  # Predicted less urgent
        elif p_urg < t_urg:
            over += 1   # Predicted more urgent
        else:
            correct += 1
    
    return SafetyMetrics(
        emergency_sensitivity=sens,
        emergency_specificity=spec,
        under_triage_rate=under / n if n > 0 else 0.0,
        over_triage_rate=over / n if n > 0 else 0.0,
        triage_accuracy=correct / n if n > 0 else 0.0,
    )
