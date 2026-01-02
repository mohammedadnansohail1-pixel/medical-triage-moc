# Medical Triage System - Project Context

## Current Phase: Phase 5 COMPLETE

## Project Overview
Two-tier AI medical triage system with LLM-powered explanations.

## Architecture
```
User Input (symptoms, age, sex)
         ↓
┌─────────────────────────────────────┐
│ Stage 1: Symptom Normalization      │ → Expand synonyms
│ Stage 2: Entity Linking (SapBERT)   │ → Match to SNOMED codes
│ Stage 3: Vectorization              │ → Binary symptom vectors
│ Stage 4: Specialty Routing          │ → Tier 1 classification
│ Stage 5: Differential Diagnosis     │ → Tier 2 within specialty
│ Stage 6: LLM Explanation            │ → Mistral 7B via Ollama
└─────────────────────────────────────┘
         ↓
Response: specialty, confidence, DDx[], explanation
```

## Completed Phases

### Phase 1: Data Pipeline ✓
- DDXPlus dataset integration (1.3M cases, 49 conditions)
- Evidence/symptom extraction and normalization

### Phase 2: Specialty Routing (Tier 1) ✓
- 6 specialties: emergency, cardiology, pulmonology, neurology, gastroenterology, general_medicine
- 90% accuracy on DDXPlus validation set
- Emergency override for critical symptom combinations

### Phase 3: Symptom Understanding ✓
- SapBERT entity linking (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
- Cosine similarity matching to SNOMED codes
- Synonym expansion for robust matching

### Phase 4: Differential Diagnosis (Tier 2) ✓
- Specialty-specific classifiers
- 95% Top-1 accuracy, 99.8% Top-3 accuracy
- Bayesian probability estimation

### Phase 5: LLM Explanation Layer ✓
- Ollama + Mistral 7B (4GB VRAM)
- Patient-friendly explanations
- Urgency classification (emergency/urgent/routine)
- Fallback mechanism for reliability
- ~1.2s generation time

## Key Files
```
backend/
├── app/
│   ├── main.py                      # FastAPI application
│   ├── api/
│   │   └── triage.py                # POST /api/v1/triage endpoint
│   └── core/
│       ├── triage_pipeline_v2.py    # Main pipeline orchestrator
│       ├── symptom_linker.py        # SapBERT entity linking
│       ├── specialty_router.py      # Tier 1 routing
│       ├── differential_agent.py    # Tier 2 DDx
│       └── explanation_generator.py # LLM explanations
├── tests/
│   ├── test_symptom_linker.py
│   ├── test_triage_pipeline.py
│   └── test_explanation_generator.py
└── data/
    ├── ddxplus/
    │   ├── release_evidences.json
    │   └── condition_model.json
    └── classifier/
        ├── model.pkl
        └── vocabulary.pkl
```

## Hardware Requirements
- GPU: NVIDIA RTX 4080 12GB (or equivalent)
- VRAM Usage: ~5.5GB (SapBERT 0.5GB + Mistral 4GB + overhead)
- Ollama running on localhost:11434

## API Usage
```bash
# Start Ollama
ollama serve &

# Start API
cd backend && uvicorn app.main:app --port 8000

# Request
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["chest pain", "shortness of breath"],
    "age": 55,
    "sex": "male"
  }'
```

## Response Format
```json
{
  "specialty": "emergency",
  "confidence": 1.0,
  "differential_diagnosis": [
    {"condition": "Unstable angina", "probability": 0.85, "rank": 1}
  ],
  "explanation": {
    "text": "Patient-friendly explanation...",
    "urgency": "emergency",
    "next_steps": ["Call 911", "..."]
  },
  "route": "EMERGENCY_OVERRIDE"
}
```

## Performance Metrics
| Metric | Value |
|--------|-------|
| Specialty Routing Accuracy | 90% |
| DDx Top-1 Accuracy | 95% |
| DDx Top-3 Accuracy | 99.8% |
| Explanation Generation | ~1.2s |
| Total Pipeline Latency | ~3-4s |

## Next Steps (Phase 6 Options)
1. **Frontend UI** - React/Next.js symptom input interface
2. **Conversation Agent** - Multi-turn symptom collection
3. **Evaluation Dashboard** - Metrics and A/B testing
4. **Production Hardening** - Docker, logging, monitoring

## Decisions Made
- Ollama over vLLM: Simpler setup, sufficient performance
- Mistral 7B over larger models: Fits in VRAM with SapBERT
- JSON format enforcement: `format: "json"` in Ollama API
- Fallback explanations: Ensure reliability when LLM fails
- Emergency override: Bypass DDx for critical symptoms

## Commands Reference
```bash
# Test pipeline
cd ~/projects/medical-triage-moc/backend
python -c "from app.core.triage_pipeline_v2 import get_triage_pipeline; ..."

# Run tests
pytest tests/ -v

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
