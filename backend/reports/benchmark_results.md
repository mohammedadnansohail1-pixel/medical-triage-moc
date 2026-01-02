# Medical Triage Evaluation Report

**Model:** Medical Triage System
**Date:** 2026-01-01 23:38
**Samples:** 90

## Safety: ❌ FAIL

| Metric | Value | Threshold |
|--------|-------|-----------|
| Emergency Sensitivity | 0.0% | ≥95% |
| Under-triage Rate | 22.2% | ≤5% |

## Tier 1: Specialty Routing

| Metric | Value |
|--------|-------|
| Accuracy | 66.7% |
| Macro F1 | 63.9% |
| Weighted F1 | 63.9% |

## Tier 2: Differential Diagnosis

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 16.7% |
| Top-3 Accuracy | 27.8% |
| MRR | 0.213 |

## Calibration

| Metric | Value |
|--------|-------|
| Brier Score | 0.3148 |
| ECE | 0.2566 |