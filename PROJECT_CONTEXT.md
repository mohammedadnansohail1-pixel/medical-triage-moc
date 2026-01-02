# Medical Triage MOC - Project Context

## Current Phase: 3 (Complete)

## Project Overview
AI-powered medical triage routing system that maps patient symptoms to appropriate medical specialties.

## Architecture
```
Patient text
    ↓
┌─────────────────────────────────────┐
│ Stage 0: Emergency Detection        │
│ Rule-based regex patterns           │
│ 100% reliable, always runs first    │
└─────────────────────────────────────┘
    ↓ (if not emergency)
┌─────────────────────────────────────┐
│ Stage 1: SapBERT Entity Linking     │
│ Patient language → Evidence codes   │
│ Model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext │
│ VRAM: 0.42 GB                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 2: XGBoost Classification     │
│ Evidence codes → Specialty          │
│ 225 features, 7 classes             │
│ Trained on DDXPlus dataset          │
└─────────────────────────────────────┘
    ↓
Specialty routing (7 specialties)
```

## Completed Work

### Phase 1: Foundation
- Project structure with FastAPI backend
- Entity extraction (symptoms, negations, duration, severity)
- Neo4j knowledge graph integration
- Docker compose setup

### Phase 2: Evaluation Framework
- 22 curated test cases across specialties
- DDXPlus benchmark integration (50K train, 5K test)
- XGBoost classifier: 99.9% on evidence codes
- LLM routing: 58% on patient text
- Ensemble combiner (KG + LLM + Rules)

### Phase 3: SapBERT Pipeline
- SapBERT entity linker for patient language → evidence codes
- Emergency detector with rule-based override
- Triage Pipeline v2 integrating all components
- API endpoint: POST /api/triage/v2
- **Results: 79.4% accuracy (+21% over LLM baseline)**

## Current Metrics

### Overall Accuracy: 79.4% (500 samples)

| Specialty | Accuracy | Notes |
|-----------|----------|-------|
| neurology | 100% | Excellent |
| dermatology | 100% | Small sample (14) |
| cardiology | 93% | Excellent |
| pulmonology | 86.8% | Good |
| emergency | 84% | Needs improvement |
| general_medicine | 64.1% | Catch-all confusion |
| gastroenterology | 53.7% | Poor - overlapping symptoms |

### Resource Usage
- VRAM: 0.42 GB (vs 8GB for LLM)
- Latency: ~100ms (vs ~2s for LLM)

## Known Issues

1. **Gastroenterology confusion (53.7%)**: Overlapping symptoms with general_medicine (nausea, stomach pain)
2. **General_medicine catch-all (64.1%)**: Too many cases routed here incorrectly
3. **DDXPlus data missing**: Only have release_evidences.json, patient files download failed

## File Structure
```
~/projects/medical-triage-moc/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── triage.py        # V1 endpoint (LLM)
│   │   │   └── triage_v2.py     # V2 endpoint (SapBERT)
│   │   ├── core/
│   │   │   ├── sapbert_linker.py
│   │   │   ├── emergency_detector.py
│   │   │   ├── triage_pipeline_v2.py
│   │   │   ├── clinical_bert_classifier.py  # Future use
│   │   │   ├── train_clinical_bert.py       # Future use
│   │   │   ├── entity_extractor.py
│   │   │   ├── ensemble.py
│   │   │   └── classifier/
│   │   │       ├── train.py
│   │   │       └── model.py
│   │   └── main.py
│   ├── data/classifier/
│   │   ├── model.pkl           # XGBoost model
│   │   ├── vocabulary.pkl      # Code/specialty mappings
│   │   ├── train_data.pkl      # 50K training samples
│   │   └── test_data.pkl       # 5K test samples
│   └── evaluation/
│       ├── test_cases.py       # 22 curated cases
│       └── runner.py
├── data/ddxplus/
│   └── release_evidences.json  # 223 evidence definitions
└── docker-compose.yml
```

## API Endpoints

### V2 (Current - SapBERT)
```bash
# Health check
GET /api/triage/v2/health

# Triage request
POST /api/triage/v2
{
  "symptoms": ["chest pain", "difficulty breathing"],
  "age": 55,
  "sex": "male"
}

# Response
{
  "specialty": "emergency",
  "confidence": 1.0,
  "urgency": "emergency",
  "matched_codes": [],
  "reasoning": ["Emergency detected: cardiac_emergency"],
  "route": "EMERGENCY_OVERRIDE",
  "is_emergency": true,
  "emergency_reason": "cardiac_emergency"
}
```

## Next Steps

### Immediate: Improve Accuracy
- Analyze gastro vs general_medicine confusion matrix
- Tune SapBERT similarity threshold
- Consider class weights in XGBoost retraining
- NO hardcoding or overfitting to test set

### Future: Phase 4
- Specialty-specific agents
- Follow-up question generation
- Diagnosis refinement within specialty

## Commands
```bash
# Activate environment
cd ~/projects/medical-triage-moc/backend
source venv/bin/activate

# Run API
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run pipeline test
PYTHONPATH=. python << 'EOF'
from pathlib import Path
from app.core.triage_pipeline_v2 import get_triage_pipeline

pipeline = get_triage_pipeline()
pipeline.load(
    Path("/home/adnan21/projects/medical-triage-moc/data/ddxplus/release_evidences.json"),
    Path("data/classifier/model.pkl"),
    Path("data/classifier/vocabulary.pkl"),
)

result = pipeline.predict(["chest pain", "shortness of breath"])
print(result)
pipeline.unload()
