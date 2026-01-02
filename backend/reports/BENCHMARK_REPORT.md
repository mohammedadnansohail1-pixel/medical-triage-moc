# Medical Triage System - Benchmark Report

**Date:** 2026-01-01
**Dataset:** DDXPlus (NeurIPS 2022)
**Samples:** 100

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **58.0%** |
| Top-2 Accuracy | 61.0% |
| Emergency Sensitivity | 100% (3/3) |
| Cardiology Accuracy | 100% (11/11) |

## Tier 1: Specialty Routing Performance

| Specialty | Accuracy | Correct | Total |
|-----------|----------|---------|-------|
| emergency | 100% | 3 | 3 |
| cardiology | 100% | 11 | 11 |
| pulmonology | 89% | 33 | 37 |
| neurology | 33% | 3 | 9 |
| general_medicine | 29% | 8 | 28 |
| gastroenterology | 0% | 0 | 9 |
| dermatology | 0% | 0 | 3 |

## Safety Analysis

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Emergency Sensitivity | 100% | ≥95% | ✅ PASS |
| Cardiology Sensitivity | 100% | ≥90% | ✅ PASS |
| Critical Miss Rate | 0% | ≤5% | ✅ PASS |

## Comparison with Published Baselines

| System | Dataset | Accuracy | Source |
|--------|---------|----------|--------|
| **This Model** | DDXPlus | **58%** | Current |
| Symptom Checkers (median) | Vignettes | 34% | JMIR 2022 |
| Ada Health | ED Study | 51% | PMC 2022 |
| DDXPlus Baseline (GTPA@1) | DDXPlus | 58% | NeurIPS 2022 |

## Strengths & Weaknesses

**Strengths (≥80% accuracy):**
- Cardiology: 100%
- Emergency: 100%  
- Pulmonology: 89%

**Needs Improvement (<50% accuracy):**
- Gastroenterology: 0%
- Dermatology: 0%
- General Medicine: 29%
- Neurology: 33%

## Methodology

Evaluation follows standards from:
- DDXPlus (NeurIPS 2022): Differential diagnosis benchmark
- MedHELM (Stanford HAI): Multi-metric evaluation
- Lancet Digital Health: Clinical AI evaluation guidelines
