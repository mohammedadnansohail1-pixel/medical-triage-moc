# PROJECT CONTEXT - Medical Triage MOC

## Current Phase
**Phase 2: Core Pipeline ✅ COMPLETE**

## Completed Work
- [x] System requirements verified (i9-13900HX, 24GB RAM, RTX 4080 12GB)
- [x] Git repository initialized
- [x] GitHub private repo: https://github.com/mohammedadnansohail1-pixel/medical-triage-moc
- [x] Project folder structure created
- [x] .env.example and .env configured
- [x] Makefile with common commands
- [x] docker-compose.yml (all services defined)
- [x] Backend Dockerfile - builds successfully
- [x] FastAPI skeleton with health + triage endpoints
- [x] Ollama + GPU tested (RTX 4080, llama3.1:8b)
- [x] LLM Provider interface (abstracted Ollama/Claude)
- [x] Entity Extractor (20 symptoms, duration, severity, negation)
- [x] Knowledge Graph Router (Neo4j queries)
- [x] Knowledge Graph seed data (8 specialties, 20 symptoms, 19 diseases)
- [x] LLM Router with chain-of-thought reasoning
- [x] Ensemble Router (KG 30% + LLM 50% + Rules 20%)
- [x] Emergency override rules
- [x] Full pipeline tested end-to-end ✅

## Test Results
| Case | Specialty | Urgency | Status |
|------|-----------|---------|--------|
| Cardiac (chest pain + SOB + sweating) | emergency | emergency | ✅ |
| Headache + nausea | gastroenterology | routine | ✅ |
| GI (stomach pain + nausea) | gastroenterology | routine | ✅ |

## Next Steps
1. [ ] Start Neo4j and test KG routing
2. [ ] Build frontend (Next.js)
3. [ ] Download DDXPlus dataset for evaluation
4. [ ] Run baseline comparisons
5. [ ] Create metrics dashboard

## Key Commands
```bash
# Start Ollama with GPU
docker run --rm --gpus all -p 11434:11434 -v triage_ollama_data:/root/.ollama ollama/ollama

# Start backend (from backend/)
source venv/bin/activate
PYTHONPATH=. uvicorn app.main:app --reload

# Test triage
curl -X POST http://localhost:8000/api/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "chest pain and shortness of breath", "age": 55, "sex": "male"}'
```

## Architecture
```
Patient Input → Entity Extraction → Ensemble Router → Response
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              Knowledge Graph       LLM Router          Rule-based
                 (30%)               (50%)               (20%)
                    │                    │                    │
                    └────────────────────┴────────────────────┘
                                         │
                              Emergency Override (if triggered)
```

## Tech Stack
| Component | Technology | Status |
|-----------|------------|--------|
| Backend | FastAPI + Python 3.12 | ✅ Working |
| LLM | Ollama + llama3.1:8b | ✅ Working |
| Entity Extraction | Regex + patterns | ✅ Working |
| Ensemble Router | KG + LLM + Rules | ✅ Working |
| Knowledge Graph | Neo4j | ⏳ Seed data ready |
| Frontend | Next.js 14 | ⏳ Pending |
| Database | PostgreSQL | ⏳ Pending |

## How to Resume
1. Read this file
2. Start Ollama: `docker run --rm --gpus all -p 11434:11434 -v triage_ollama_data:/root/.ollama ollama/ollama`
3. Start backend: `cd backend && source venv/bin/activate && PYTHONPATH=. uvicorn app.main:app --reload`
4. Check "Next Steps"
