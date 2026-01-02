"""Report Generator for Medical Triage Benchmark."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from app.evaluation.benchmark import BenchmarkResult


def generate_markdown_report(
    result: BenchmarkResult,
    output_path: Path,
    model_name: str = "Medical Triage System",
) -> Path:
    """Generate Markdown report."""
    r = []
    r.append("# Medical Triage Evaluation Report\n")
    r.append(f"**Model:** {model_name}")
    r.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    r.append(f"**Samples:** {result.n_samples}\n")
    
    safe = result.safety.passes_safety_threshold()
    r.append(f"## Safety: {'✅ PASS' if safe['overall_safe'] else '❌ FAIL'}\n")
    r.append("| Metric | Value | Threshold |")
    r.append("|--------|-------|-----------|")
    r.append(f"| Emergency Sensitivity | {result.safety.emergency_sensitivity:.1%} | ≥95% |")
    r.append(f"| Under-triage Rate | {result.safety.under_triage_rate:.1%} | ≤5% |\n")
    
    r.append("## Tier 1: Specialty Routing\n")
    r.append("| Metric | Value |")
    r.append("|--------|-------|")
    r.append(f"| Accuracy | {result.classification.accuracy:.1%} |")
    r.append(f"| Macro F1 | {result.classification.macro_f1:.1%} |")
    r.append(f"| Weighted F1 | {result.classification.weighted_f1:.1%} |\n")
    
    r.append("## Tier 2: Differential Diagnosis\n")
    r.append("| Metric | Value |")
    r.append("|--------|-------|")
    r.append(f"| Top-1 Accuracy | {result.ranking.top_1_accuracy:.1%} |")
    r.append(f"| Top-3 Accuracy | {result.ranking.top_3_accuracy:.1%} |")
    r.append(f"| MRR | {result.ranking.mrr:.3f} |\n")
    
    r.append("## Calibration\n")
    r.append("| Metric | Value |")
    r.append("|--------|-------|")
    r.append(f"| Brier Score | {result.calibration.brier_score:.4f} |")
    r.append(f"| ECE | {result.calibration.ece:.4f} |")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(r))
    return output_path


def generate_json_report(
    result: BenchmarkResult,
    output_path: Path,
) -> Path:
    """Generate JSON report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": result.to_dict(),
        "safety_check": result.safety.passes_safety_threshold(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return output_path
