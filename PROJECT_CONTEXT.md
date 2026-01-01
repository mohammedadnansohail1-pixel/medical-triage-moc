# PROJECT CONTEXT - Medical Triage MOC

## Current Phase
**Phase 1: Backend Skeleton ✅ COMPLETE**

## Completed Work
- [x] System requirements verified (i9-13900HX, 24GB RAM, RTX 4080 12GB)
- [x] Git repository initialized
- [x] GitHub private repo: https://github.com/mohammedadnansohail1-pixel/medical-triage-moc
- [x] Project folder structure created
- [x] .env.example and .env configured
- [x] Makefile with common commands
- [x] docker-compose.yml (all services defined)
- [x] Backend Dockerfile
- [x] FastAPI skeleton with health + triage endpoints
- [x] Pydantic models with validation
- [x] Structured logging (structlog)
- [x] Local test passed: /health and /api/triage working

## Next Steps
1. [ ] Test Docker build for backend
2. [ ] Set up Ollama with Mistral-7B
3. [ ] Create LLM provider interface (abstracted)
4. [ ] Create entity extractor module
5. [ ] Create knowledge graph router
6. [ ] Download DDXPlus dataset for evaluation

## Decisions Made
- **LLM Strategy**: Local first (Ollama + Mistral-7B), abstracted interface
- **Architecture**: Portable Docker Compose, env var config
- **Resource Limits**: ~3GB RAM when running
- **Git Protocol**: HTTPS
- **Python Path**: Run with `PYTHONPATH=.` for local dev

## Key Files
| File | Purpose |
|------|---------|
| docker-compose.yml | All services |
| .env | Configuration (git ignored) |
| Makefile | CLI commands |
| backend/app/main.py | FastAPI entry point |
| backend/app/config.py | Settings with validation |
| backend/app/api/routes/triage.py | Triage endpoint |

## Tech Stack
| Component | Technology | Status |
|-----------|------------|--------|
| Backend | FastAPI + Python 3.12 | ✅ Working |
| Frontend | Next.js 14 | ⏳ Pending |
| LLM (Local) | Ollama + Mistral-7B | ⏳ Pending |
| LLM (Cloud) | Claude API (fallback) | ⏳ Pending |
| Knowledge Graph | Neo4j | ⏳ Pending |
| Database | PostgreSQL | ⏳ Pending |
| Cache | Redis | ⏳ Pending |

## Local Dev Commands
```bash
cd ~/projects/medical-triage-moc/backend
source venv/bin/activate
PYTHONPATH=. uvicorn app.main:app --reload
```

## How to Resume
1. Read this file
2. Check "Next Steps"
3. Run local dev commands to verify setup
