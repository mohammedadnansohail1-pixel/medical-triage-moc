# PROJECT CONTEXT - Medical Triage MOC

## Current Phase
**Phase 3: Evaluation ✅ COMPLETE**

## Evaluation Results (22 test cases)
| Metric | Score |
|--------|-------|
| Specialty Accuracy | **86.4%** |
| Urgency Accuracy | 72.7% |
| Emergency Sensitivity | 71.4% |
| Avg Latency | 2306ms |
| Avg Confidence | 0.77 |

### Per-Specialty Accuracy
- cardiology: 100%
- dermatology: 100%
- emergency: 100%
- gastroenterology: 100%
- general_medicine: 100%
- neurology: 75%
- orthopedics: 66.7%
- pulmonology: 66.7%

### Known Issues
- 2 timeout errors (PULM-002, ORTH-003)
- Migraine misclassified as GI (nausea symptom)

## Completed Work
- [x] Full ensemble working (KG + LLM + Rules)
- [x] Knowledge Graph seeded (8 specialties, 20 symptoms, 19 diseases)
- [x] 22 curated test cases
- [x] Evaluation runner with metrics
- [x] 86.4% specialty accuracy achieved

## Next Steps (Priority Order)
1. [ ] Fix timeout errors (increase LLM timeout)
2. [ ] Improve migraine detection (add photophobia pattern)
3. [ ] Baseline comparison (rule-only, LLM-only)
4. [ ] Push to GitHub
5. [ ] Frontend (optional for MOC)

## How to Run

### Start Services
```bash
cd ~/projects/medical-triage-moc
docker compose up -d neo4j ollama
```

### Start Backend
```bash
cd backend
source venv/bin/activate
PYTHONPATH=. uvicorn app.main:app --reload
```

### Run Evaluation
```bash
cd backend
PYTHONPATH=. python -m evaluation.runner
```

### Test Single Case
```bash
curl -X POST http://localhost:8000/api/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "chest pain and shortness of breath", "age": 55, "sex": "male"}'
```

## Architecture
```
Input → Entity Extractor → Ensemble Router → Response
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         Knowledge Graph    LLM Router    Rule-based
            (30%)             (50%)          (20%)
              │                │                │
              └────────────────┴────────────────┘
                               │
                    Emergency Override
```

## Files Changed
- backend/app/core/entity_extractor.py
- backend/app/core/llm_provider.py
- backend/app/core/llm_router.py
- backend/app/core/knowledge_graph.py
- backend/app/core/ensemble.py
- backend/app/api/routes/triage.py
- backend/evaluation/test_cases.py
- backend/evaluation/runner.py
- knowledge_graph/seed_data.cypher

## GitHub Repo
https://github.com/mohammedadnansohail1-pixel/medical-triage-moc
