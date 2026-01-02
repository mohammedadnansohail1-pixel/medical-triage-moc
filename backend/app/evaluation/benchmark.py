"""Benchmark Runner for Medical Triage System."""

import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from app.evaluation.metrics import (
    compute_classification_metrics,
    compute_ranking_metrics,
    ClassificationMetrics,
    RankingMetrics,
)
from app.evaluation.calibration import compute_calibration, CalibrationResult
from app.evaluation.safety import compute_safety_metrics, SafetyMetrics


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    n_samples: int
    runtime_seconds: float
    classification: ClassificationMetrics
    ranking: RankingMetrics
    calibration: CalibrationResult
    safety: SafetyMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "classification": self.classification.to_dict(),
            "ranking": self.ranking.to_dict(),
            "calibration": self.calibration.to_dict(),
            "safety": self.safety.to_dict(),
        }

    def print_summary(self):
        print("=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Samples: {self.n_samples}")
        print(f"Runtime: {self.runtime_seconds:.2f}s")
        print(f"\n-- Tier 1: Specialty Routing --")
        print(f"Accuracy: {self.classification.accuracy:.1%}")
        print(f"Macro F1: {self.classification.macro_f1:.1%}")
        print(f"\n-- Tier 2: Differential Diagnosis --")
        print(f"Top-1: {self.ranking.top_1_accuracy:.1%}")
        print(f"Top-3: {self.ranking.top_3_accuracy:.1%}")
        print(f"MRR: {self.ranking.mrr:.3f}")
        print(f"\n-- Calibration --")
        print(f"Brier: {self.calibration.brier_score:.4f}")
        print(f"ECE: {self.calibration.ece:.4f}")
        print(f"\n-- Safety --")
        print(f"Emergency Sens: {self.safety.emergency_sensitivity:.1%}")
        print(f"Under-triage: {self.safety.under_triage_rate:.1%}")
        safe = self.safety.passes_safety_threshold()["overall_safe"]
        print(f"Safe: {'✅ PASS' if safe else '❌ FAIL'}")
        print("=" * 50)


class BenchmarkRunner:
    """Run benchmark on pipeline."""

    def __init__(self, pipeline, specialty_labels: List[str]):
        self.pipeline = pipeline
        self.specialty_labels = specialty_labels

    def run(
        self,
        samples: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> BenchmarkResult:
        """Run benchmark on samples."""
        n = len(samples)
        
        spec_true, spec_pred, confs = [], [], []
        diag_true, diag_ranked = [], []
        
        start = time.time()
        
        for i, sample in enumerate(samples):
            if verbose and i % 100 == 0:
                print(f"  Processing {i}/{n}...")
            
            result = self.pipeline.predict(
                symptoms=sample["symptoms"],
                age=sample.get("age"),
                sex=sample.get("sex"),
            )
            
            spec_true.append(sample["specialty"])
            spec_pred.append(result["specialty"])
            confs.append(result["confidence"])
            
            diag_true.append(sample["diagnosis"])
            ddx = result.get("differential_diagnosis", [])
            ranked = [(d["condition"], d["probability"]) for d in ddx]
            diag_ranked.append(ranked)
        
        elapsed = time.time() - start
        
        return BenchmarkResult(
            n_samples=n,
            runtime_seconds=elapsed,
            classification=compute_classification_metrics(spec_true, spec_pred, self.specialty_labels),
            ranking=compute_ranking_metrics(diag_true, diag_ranked),
            calibration=compute_calibration(spec_true, spec_pred, confs),
            safety=compute_safety_metrics(spec_true, spec_pred),
        )
