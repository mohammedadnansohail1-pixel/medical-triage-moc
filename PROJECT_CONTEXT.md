# Medical Triage System - Project Context

## Current Phase: PRODUCTION READY ✅

## Project Overview
Two-tier AI medical triage system with LLM-powered explanations.

## Final Metrics

| Metric | Value |
|--------|-------|
| DDXPlus Accuracy | 99.90% |
| Natural Language Accuracy | 100% (5/5) |
| Emergency Detection | 100% |
| Test Suite | 51/51 passed |
| API Latency (no LLM) | <100ms |
| API Latency (with LLM) | ~1.3s |
| Docker Image Size | 4.55GB |

## Architecture
```
User Input (symptoms, age, sex)
         ↓
┌─────────────────────────────────────┐
│ Stage 1: Symptom Normalization      │ → Expand synonyms
│ Stage 2: Emergency Detection        │ → Rule-based (100% reliable)
│ Stage 2.5: Specialty Rules          │ → Dermatology/Gastro override
│ Stage 3: SapBERT Entity Linking     │ → Match to SNOMED codes
│ Stage 4: XGBoost Classification     │ → 7-class specialty routing
│ Stage 5: Differential Diagnosis     │ → Bayesian within specialty
│ Stage 6: LLM Explanation            │ → llama3.1:8b via Ollama
└─────────────────────────────────────┘
         ↓
Response: specialty, confidence, DDx[], explanation, route
```

## Routes
- `EMERGENCY_OVERRIDE` - Rule-based emergency (100% confidence)
- `RULE_OVERRIDE` - Keyword-based specialty (dermatology, gastro)
- `ML_CLASSIFICATION` - XGBoost prediction
- `DEFAULT_FALLBACK` - No codes matched

## Key Files
```
medical-triage-moc/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app
│   │   ├── api/triage.py                # POST /api/v1/triage
│   │   └── core/
│   │       ├── triage_pipeline_v2.py    # Main orchestrator
│   │       ├── emergency_detector.py    # Rule-based safety
│   │       ├── symptom_normalizer.py    # Text preprocessing
│   │       ├── sapbert_linker.py        # Medical NER
│   │       ├── specialty_agent.py       # DDx generation
│   │       └── explanation_generator.py # LLM (llama3.1:8b)
│   ├── tests/
│   │   └── test_api_comprehensive.py    # 51 tests
│   ├── data/classifier/
│   │   ├── model.pkl                    # XGBoost (99.9%)
│   │   └── vocabulary.pkl               # 223 codes, 7 specialties
│   ├── Dockerfile
│   └── requirements.txt
├── data/ddxplus/
│   ├── release_evidences.json
│   └── condition_model.json             # Includes dermatology
├── docker-compose.yml                   # GPU + Ollama
├── docker-compose.cpu.yml               # CPU only
└── docs/
    ├── METRICS_ANALYSIS.md
    └── TECHNICAL_ARCHITECTURE.md
```

## Commands
```bash
# Local development
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/test_api_comprehensive.py -v

# Docker (CPU only, no LLM)
docker-compose -f docker-compose.cpu.yml up -d

# Docker (GPU + Ollama)
docker-compose up -d

# Build image
docker build -t medical-triage-api:latest -f backend/Dockerfile backend/
```

## API Usage
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Triage request
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["cough", "fever"],
    "age": 35,
    "sex": "male",
    "include_explanation": true
  }'
```

## Completed Work
- [x] Phase 1: Data Pipeline (DDXPlus integration)
- [x] Phase 2: Specialty Routing (XGBoost 99.9%)
- [x] Phase 3: Symptom Understanding (SapBERT)
- [x] Phase 4: Differential Diagnosis (Bayesian)
- [x] Phase 5: LLM Explanations (llama3.1:8b)
- [x] Phase 6: Testing (51 tests) + Docker

## Decisions Made
- XGBoost over ClinicalBERT: Better accuracy/latency tradeoff
- Rule-based emergency: 100% reliable, no ML uncertainty
- Specialty rules: Fix DDXPlus data gaps (dermatology)
- llama3.1:8b over mistral:7b: Available locally
- Docker CPU-only option: For deployments without GPU

## Known Limitations
1. DDXPlus has no real dermatology conditions (rule-based fallback)
2. 49 conditions only (synthetic dataset)
3. English only
4. No image analysis

## Future Improvements
1. Frontend UI (React/Next.js)
2. Multi-turn conversation agent
3. Real clinical data training
4. Multi-language support
