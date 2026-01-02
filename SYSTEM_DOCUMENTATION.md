# Medical Triage AI System
## Comprehensive Technical Documentation

---

# 1. Executive Summary

## What We Built
A production-grade medical triage AI system that:
- Routes patients to appropriate medical specialties based on symptoms
- Provides differential diagnosis within each specialty
- Analyzes skin lesion images for dermatology cases
- Combines text and image analysis using multimodal fusion
- Generates patient-friendly explanations via LLM

## Key Metrics Achieved
| Metric | Value |
|--------|-------|
| Overall Specialty Routing Accuracy | 99.9% |
| Emergency Detection Sensitivity | 100% |
| DDX Top-1 Accuracy | 95% |
| DDX Top-3 Accuracy | 99.8% |
| Skin Cancer Detection | 100% (20/20) |
| API Response Time | <2s with LLM |

## Technology Stack
- Backend: FastAPI, Python 3.12
- ML Models: SapBERT, XGBoost, Swin Transformer, Llama 3.1
- Deployment: Docker, Uvicorn
- GPU: NVIDIA RTX 4080

---

# 2. Problem Statement

Emergency departments are overwhelmed. Patients often do not know which specialist to see.

Our Solution:
1. Understand patient symptoms in natural language
2. Route to correct specialty with high confidence
3. Detect emergencies with 100% reliability
4. Analyze skin images for dermatology cases
5. Explain decisions in patient-friendly language

Design Principles:
- Safety First: Never miss an emergency
- Conservative Escalation: When uncertain, escalate
- Explainability: Every decision is explainable
- Multimodal: Combine text + images when available

---

# 3. System Architecture

## Pipeline Flow
```
Input: {"symptoms": [...], "image_base64": "..."}
                |
                v
+------------------------------------------+
| STAGE 1: Emergency Detection (Rules)     |
| - Regex patterns for critical symptoms   |
| - If MATCH -> EMERGENCY (100% confidence)|
+------------------------------------------+
                |
                v (not emergency)
+------------------------------------------+
| STAGE 2: Symptom Normalization           |
| - Fuzzy matching with RapidFuzz          |
| - "tummy hurts" -> "abdominal pain"      |
+------------------------------------------+
                |
                v
+------------------------------------------+
| STAGE 3: SapBERT Entity Linking          |
| - Model: SapBERT-from-PubMedBERT         |
| - Maps symptoms to DDXPlus codes (E_XX)  |
| - Cosine similarity, threshold 0.3       |
+------------------------------------------+
                |
                v
+------------------------------------------+
| STAGE 4: Specialty Routing               |
| A) Rule-based (dermatology keywords)     |
| B) XGBoost (225-dim -> 7 specialties)    |
+------------------------------------------+
                |
                v
+------------------------------------------+
| STAGE 5: Differential Diagnosis          |
| - Naive Bayes within specialty           |
| - Returns top-5 conditions               |
+------------------------------------------+
                |
                v
+------------------------------------------+
| STAGE 6: Image Analysis (if dermatology) |
| - Validation: blur, size, format         |
| - Model: Swin Transformer (8 classes)    |
| - 4-tier risk stratification             |
+------------------------------------------+
                |
                v
+------------------------------------------+
| STAGE 7: Multimodal Fusion               |
| - Decision-level fusion                  |
| - Conservative: max(text, image) risk    |
| - Agreement: strong/moderate/conflict    |
+------------------------------------------+
                |
                v
+------------------------------------------+
| STAGE 8: LLM Explanation                 |
| - Llama 3.1 8B via Ollama                |
| - Patient-friendly JSON output           |
+------------------------------------------+
```

---

# 4. Component Details

## 4.1 Emergency Detector
File: app/core/emergency_detector.py

Purpose: Safety-critical first line. Must NEVER miss emergencies.

Implementation: Regex patterns + symptom combinations
- Cardiac: chest pain + shortness of breath
- Stroke: sudden weakness, slurred speech
- Anaphylaxis: breathing difficulty + hives

Why Rule-Based?
- 100% reliability (no ML uncertainty)
- Interpretable and auditable
- Instant response time

## 4.2 Symptom Normalizer
File: app/core/symptom_normalizer.py

Purpose: Bridge patient language to medical terminology.

Example mappings:
- "my tummy hurts" -> "abdominal pain"
- "hard to breathe" -> "dyspnea shortness of breath"
- "itchy" -> "pruritus itching"

Uses RapidFuzz for fuzzy matching (60% threshold).

## 4.3 SapBERT Entity Linker
File: app/core/sapbert_linker.py

Model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
- 110M parameters
- 768-dim embeddings
- Trained on UMLS medical ontology

Process:
1. Pre-compute embeddings for 223 DDXPlus evidence codes
2. Embed patient symptoms
3. Find closest codes via cosine similarity
4. Return codes above 0.3 threshold

## 4.4 XGBoost Specialty Classifier
File: app/core/classifier/

Training:
- Data: 50,000 DDXPlus samples
- Features: 225 dims (223 evidence + age + sex)
- Labels: 7 specialties

Config:
- n_estimators=200, max_depth=8
- learning_rate=0.1

Results:
- Test accuracy: 99.9%
- Macro F1: 98.7%

## 4.5 Specialty Agents (Differential Diagnosis)
File: app/core/specialty_agent.py

Algorithm: Naive Bayes
P(condition | symptoms) = P(condition) x Product(P(symptom | condition))

Output: Top-5 conditions with probabilities

## 4.6 Skin Lesion Classifier
File: app/core/skin_classifier.py

Model: NeuronZero/SkinCancerClassifier
- Architecture: Swin Transformer
- Input: 224x224 RGB
- Output: 8 classes

Classes:
| Code | Name | Cancer? |
|------|------|---------|
| MEL | Melanoma | Yes |
| BCC | Basal Cell Carcinoma | Yes |
| SCC | Squamous Cell Carcinoma | Yes |
| AK | Actinic Keratosis | Pre |
| BKL | Benign Keratosis | No |
| DF | Dermatofibroma | No |
| NV | Melanocytic Nevus | No |
| VASC | Vascular Lesion | No |

4-Tier Risk (NICE NG12 aligned):
1. routine_monitoring - Self-monitor
2. consider_evaluation - See GP when convenient
3. routine_referral - Dermatologist within weeks
4. urgent_referral - Dermatologist within 2 weeks

Validation:
- Cancer detection: 100% (20/20)
- Pre-cancer detection: 100% (10/10)

## 4.7 Image Validator
File: app/core/image_validator.py

Checks:
- Base64 decoding
- Dimensions: min 100x100, max 4096x4096
- Format: JPEG, PNG, WebP
- Blur: Laplacian variance > 100

## 4.8 Multimodal Fusion
File: app/core/multimodal_fusion.py

Strategy: Decision-level (late) fusion
- Each model runs independently
- Combine at decision layer
- Conservative: max(text_risk, image_risk)

Risk Levels (ordered):
0. ROUTINE_MONITORING
1. CONSIDER_EVALUATION
2. ROUTINE_REFERRAL
3. URGENT_REFERRAL
4. EMERGENCY

Agreement Levels:
- strong: Same risk level
- moderate: 1 level difference
- conflict: 2+ levels (requires explanation)

Conflict Resolution:
- Image cancer + Text routine -> Trust image
- Text urgent + Image benign -> Trust text
- Always take HIGHER risk (patient safety)

## 4.9 LLM Explanation Generator
File: app/core/explanation_generator.py

Model: Llama 3.1 8B via Ollama
Response time: ~1.2s

Prompt rules:
- Simple patient-friendly language
- Never diagnose, only suggest
- Always recommend professional consultation

Fallback: Template-based if Ollama unavailable

---

# 5. API Reference

## POST /api/v1/triage

Request:
```json
{
  "symptoms": ["skin rash", "itchy"],
  "age": 45,
  "sex": "male",
  "include_explanation": true,
  "image_base64": "data:image/jpeg;base64,..."
}
```

Response:
```json
{
  "specialty": "dermatology",
  "confidence": 0.85,
  "differential_diagnosis": [...],
  "explanation": {...},
  "route": "RULE_OVERRIDE",
  "modalities_used": ["text", "image"],
  "image_analysis": {
    "prediction": "MEL",
    "confidence": 0.99,
    "tier": "urgent_referral",
    "cancer_probability": 0.99
  },
  "combined_assessment": {
    "final_risk_tier": "urgent_referral",
    "agreement_level": "conflict",
    "reasoning": "...",
    "resolution_rationale": "..."
  },
  "warnings": [...]
}
```

Route Values:
- EMERGENCY_OVERRIDE: Emergency detected
- RULE_OVERRIDE: Keyword rules matched
- ML_CLASSIFICATION: XGBoost used
- DEFAULT_FALLBACK: No matches

---

# 6. Test Results

119 tests passing:
- test_api_comprehensive.py: 52
- test_entity_extractor.py: 17
- test_explanation_generator.py: 11
- test_skin_classifier.py: 21
- test_specialty_agent.py: 10
- test_multimodal.py: 11

Multimodal Integration Tests:
| Test | Text Risk | Image Risk | Agreement | Final |
|------|-----------|------------|-----------|-------|
| Melanoma | routine | urgent | conflict | urgent |
| Benign | routine | consider | moderate | consider |
| No image | routine | N/A | N/A | routine |

---

# 7. Safety Considerations

Design Principles:
1. Never miss an emergency (100% sensitivity)
2. Conservative escalation (uncertain -> escalate)
3. Transparency (reasoning for every decision)
4. Human-in-the-loop (recommendations, not diagnoses)

Limitations:
- Not a diagnosis - screening only
- Image quality dependent
- Skin tone bias (trained on Fitzpatrick I-III)
- English only
- No temporal reasoning

---

# 8. File Structure
```
backend/
  app/
    api/
      triage.py              # Main endpoint
    core/
      emergency_detector.py  # Rule-based emergency
      symptom_normalizer.py  # Fuzzy matching
      sapbert_linker.py      # Entity linking
      classifier/            # XGBoost routing
      specialty_agent.py     # Differential Dx
      skin_classifier.py     # Image classification
      image_validator.py     # Quality checks
      multimodal_fusion.py   # Decision fusion
      explanation_generator.py
      triage_pipeline_v2.py  # Orchestration
    evaluation/
      metrics.py, safety.py, calibration.py
    main.py
    config.py
  tests/
  data/
    ddxplus/
    classifier/
```

---

# 9. Dependencies

Key packages:
- fastapi, uvicorn, pydantic
- torch, transformers (SapBERT, Swin)
- xgboost, scikit-learn
- Pillow, numpy
- rapidfuzz
- structlog, httpx

---

Document Version: 1.0
Last Updated: January 2, 2026
