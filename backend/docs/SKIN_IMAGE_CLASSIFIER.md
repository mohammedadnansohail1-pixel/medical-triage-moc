# Skin Image Classification

## Overview

AI-powered skin lesion classification using Swin Transformer model with 4-tier risk stratification aligned with NICE NG12 clinical pathways.

## Model

- **Model**: NeuronZero/SkinCancerClassifier
- **Architecture**: Swin Transformer
- **Parameters**: 86.8M
- **Classes**: 8 (MEL, BCC, SCC, AK, BKL, DF, NV, VASC)

## 4-Tier Risk Stratification

| Tier | Color | Criteria | Recommended Action |
|------|-------|----------|-------------------|
| ROUTINE_MONITORING | 🟢 Green | Benign >85%, cancer <5% | Self-monitor for changes |
| CONSIDER_EVALUATION | 🟡 Yellow | Benign 70-85%, cancer 5-15%, AK, or MEL >3% | Consider GP visit if concerned |
| ROUTINE_REFERRAL | 🟠 Orange | BCC predicted, or cancer 15-30% | Schedule dermatology appointment |
| URGENT_REFERRAL | 🔴 Red | MEL/SCC predicted, or cancer >30% | 2-week suspected cancer pathway |

## Validation Results

Tested on 70 HAM10000 dermoscopic images:

| Metric | Result |
|--------|--------|
| Cancer detection (MEL/BCC/SCC) | 100% (20/20) |
| Pre-cancer detection (AK) | 100% (10/10) |
| Benign correctly reassured | 82.5% (33/40) |

**No cancers missed** - all cancers flagged for evaluation (none in green tier).

## API Endpoints

### POST /api/v1/triage/image

Multipart form upload for skin lesion analysis.

**Parameters**:
- `image` (file, required): JPEG or PNG image
- `symptoms` (string, optional): Comma-separated symptoms
- `age` (int, optional): Patient age (0-120)
- `sex` (string, optional): "male" or "female"

### POST /api/v1/triage/image/base64

JSON request with base64-encoded image.

**Request body**:
```json
{
  "image_base64": "base64_encoded_image_data",
  "symptoms": "itching, growing",
  "age": 55,
  "sex": "male"
}
```

### Response Example
```json
{
  "specialty": "dermatology",
  "route": "IMAGE_CLASSIFICATION",
  "image_analysis": {
    "prediction": "NV",
    "prediction_label": "Melanocytic Nevus",
    "confidence": 0.92,
    "description": "Common mole, usually benign",
    "probability_summary": {
      "benign": 0.95,
      "precancer": 0.01,
      "cancer_total": 0.04,
      "melanoma": 0.03
    },
    "risk_assessment": {
      "tier": "routine_monitoring",
      "tier_display": "Routine Monitoring",
      "color": "green",
      "timeframe": "Self-monitor",
      "message": "This appears to be a benign skin lesion...",
      "action": "Continue regular skin self-checks",
      "reasons": ["High confidence benign lesion (92.0%)"]
    },
    "disclaimer": "This is an AI-assisted screening tool..."
  },
  "symptoms_provided": [],
  "patient_age": null,
  "patient_sex": null
}
```

## Design Decisions

1. **Melanoma-specific threshold (3%)**: Even low melanoma probability triggers yellow tier due to its danger
2. **Tiered messaging**: Reduces unnecessary anxiety compared to binary "cancer/not cancer"
3. **NICE NG12 alignment**: Matches clinical referral pathways (urgent vs routine)
4. **Context adjustment**: Age >50 or concerning symptoms can upgrade tier

## Limitations

- Trained primarily on lighter skin tones (Fitzpatrick I-III)
- Requires dermoscopic-quality images for best results
- NOT a medical diagnosis - always consult a dermatologist
- Model may confuse AK with SCC (both require evaluation anyway)
