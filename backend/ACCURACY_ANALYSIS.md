
# Accuracy Analysis Summary
Generated: 2026-01-02 00:41

## Root Cause Analysis

### Problem
- XGBoost model: 99.9% accuracy on test data
- Real DDXPlus inference: 58% accuracy
- Gap: 42%

### Root Cause Identified
DDXPlus training data has MISALIGNED symptom-specialty mappings:

1. **Dermatology** (was 0%):
   - Training data uses: liver cirrhosis, calcium blockers, weight gain codes
   - Actual rash codes (E_130, E_129) mapped to emergency/general_medicine
   - No DDXPlus condition uses real dermatology symptoms

2. **Gastroenterology** (was 0%):
   - Similar issue - GI-specific codes not properly mapped

3. **SapBERT** works correctly:
   - "rash" → E_130 (correct!)
   - But E_130 → emergency in XGBoost (wrong training data)

## Solution Implemented

Added rule-based specialty override in triage_pipeline_v2.py:
- Stage 2.5: SYMPTOM_SPECIALTY_RULES fires BEFORE XGBoost
- Dermatology: keyword match (rash, skin, itch, etc.) → 85% confidence
- Gastroenterology: keyword + evidence code match → 80% confidence

## Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Overall | 58% | 69.5% | +11.5% |
| Dermatology | 0% | 67% | +67% |
| Neurology | 33% | 86% | +53% |
| Gastroenterology | 0% | 58% | +58% |
| General Medicine | 29% | 51% | +22% |

## Files Modified
- backend/app/core/triage_pipeline_v2.py
  - Added SYMPTOM_SPECIALTY_RULES constant
  - Added _apply_specialty_rules() function
  - Modified predict() to use rules before XGBoost

## Remaining Issues
1. General medicine at 51% - catch-all category with high variance
2. Emergency ML predictions blocked (safety feature, not a bug)
3. Some pulmonology/cardiology slight regression from rule interference

## Next Steps for Further Improvement
1. Expand rule coverage for other weak specialties
2. Consider retraining XGBoost on SapBERT-recovered codes
3. Add confidence-based fallback to general_medicine
4. Tune rule thresholds based on more testing
