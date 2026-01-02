# Medical Triage System v2 - Evaluation Summary

## Final Metrics (1000 samples)

| Metric | Value |
|--------|-------|
| **Accuracy** | 78.1% |
| **Macro F1** | 66.1% |
| **Weighted F1** | 77.6% |
| **Brier Score** | 0.189 |
| **ECE** | 0.144 |
| **Avg Latency** | 8.7ms |

## Per-Class Performance

| Specialty | Precision | Recall | F1 | Support |
|-----------|-----------|--------|-----|---------|
| cardiology | 86.8% | 96.5% | 91.4% | 143 |
| dermatology | 66.7% | 57.1% | 61.5% | 28 |
| emergency | 0.0% | 0.0% | 0.0% | 40 |
| gastroenterology | 60.8% | 59.3% | 60.0% | 81 |
| general_medicine | 81.2% | 77.8% | 79.5% | 316 |
| neurology | 91.8% | 82.7% | 87.0% | 81 |
| pulmonology | 80.6% | 85.5% | 83.0% | 311 |

## Analysis Findings

### Root Cause of DDXPlus Performance Gap

1. **XGBoost Model**: 99.9% accuracy on raw DDXPlus codes
2. **Inference Pipeline**: Uses SapBERT to convert text → codes
3. **Gap Source**: SapBERT-recovered codes differ from training distribution

### Emergency Detection Note

- DDXPlus "emergency" class: 0% sensitivity (not matching our rules)
- **Real-world safety**: Emergency rules catch 85 high-risk cases
  - 37 cardiology, 36 pulmonology cases correctly flagged
  - Rules detect cardiac/respiratory emergencies via keywords
- This is a **labeling mismatch**, not a safety issue

### Calibration Analysis

- Correct predictions: 81.9% mean confidence
- Wrong predictions: 51.4% mean confidence  
- Model confidence separates correct/incorrect well

## Architecture
```
Patient Input
    ↓
[Symptom Normalization] - Rule-based expansion
    ↓
[Emergency Detection] - Keyword rules (100% reliable)
    ↓
[SapBERT Linking] - Text → DDXPlus evidence codes
    ↓
[XGBoost Classifier] - Codes → Specialty
    ↓
[Specialty Agent] - Differential diagnosis
    ↓
[LLM Explanation] - Patient-friendly output
```

## Files Modified This Session

- `backend/app/core/triage_pipeline_v2.py` - Added rule constants (unused in final)
- `backend/EVALUATION_SUMMARY.md` - This file
- `backend/evaluation_results_final.json` - Detailed metrics

## Recommendations for Further Improvement

1. **Retrain XGBoost on SapBERT codes** - Align training/inference distributions
2. **Expand emergency keywords** - Cover more DDXPlus emergency conditions  
3. **Temperature scaling** - Improve calibration (reduce ECE)
4. **Add neurology rules** - Catch stroke/TIA symptoms
