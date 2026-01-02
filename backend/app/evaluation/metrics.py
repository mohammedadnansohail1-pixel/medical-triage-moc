"""Core Evaluation Metrics for Medical Triage System."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ClassificationMetrics:
    """Classification metrics for Tier 1 specialty routing."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    class_labels: Optional[List[str]] = None
    auroc_macro: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_recall": self.per_class_recall,
        }


@dataclass
class RankingMetrics:
    """Ranking metrics for Tier 2 differential diagnosis."""
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mrr: float
    coverage_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_1_accuracy": self.top_1_accuracy,
            "top_3_accuracy": self.top_3_accuracy,
            "top_5_accuracy": self.top_5_accuracy,
            "mrr": self.mrr,
        }


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    class_labels: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """Compute classification metrics."""
    if class_labels is None:
        class_labels = sorted(set(y_true) | set(y_pred))
    
    n_classes = len(class_labels)
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    
    # Build confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            confusion[label_to_idx[true_label], label_to_idx[pred_label]] += 1
    
    # Per-class metrics
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    per_class_support = {}
    
    for idx, label in enumerate(class_labels):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        support = int(confusion[idx, :].sum())
        per_class_support[label] = support
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_precision[label] = float(precision)
        per_class_recall[label] = float(recall)
        per_class_f1[label] = float(f1)
    
    # Aggregates
    supports = np.array([per_class_support[l] for l in class_labels])
    macro_precision = np.mean(list(per_class_precision.values()))
    macro_recall = np.mean(list(per_class_recall.values()))
    macro_f1 = np.mean(list(per_class_f1.values()))
    weighted_f1 = np.average(list(per_class_f1.values()), weights=supports) if supports.sum() > 0 else 0.0
    accuracy = np.trace(confusion) / confusion.sum() if confusion.sum() > 0 else 0.0
    
    return ClassificationMetrics(
        accuracy=float(accuracy),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        confusion_matrix=confusion,
        class_labels=class_labels,
    )


def compute_ranking_metrics(
    y_true: List[str],
    y_pred_ranked: List[List[Tuple[str, float]]],
) -> RankingMetrics:
    """Compute ranking metrics for differential diagnosis."""
    n = len(y_true)
    if n == 0:
        return RankingMetrics(0, 0, 0, 0, 0)
    
    top1 = top3 = top5 = covered = 0
    rr_sum = 0.0
    
    for true_label, preds in zip(y_true, y_pred_ranked):
        pred_labels = [p[0] for p in preds]
        try:
            rank = pred_labels.index(true_label) + 1
            covered += 1
            rr_sum += 1.0 / rank
            if rank == 1: top1 += 1
            if rank <= 3: top3 += 1
            if rank <= 5: top5 += 1
        except ValueError:
            pass
    
    return RankingMetrics(
        top_1_accuracy=top1 / n,
        top_3_accuracy=top3 / n,
        top_5_accuracy=top5 / n,
        mrr=rr_sum / n,
        coverage_rate=covered / n,
    )
