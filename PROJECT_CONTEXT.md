# PROJECT CONTEXT - Medical Triage MOC

## Current Phase
**Phase 1: Project Setup**

## Completed Work
- [x] System requirements verified (i9-13900HX, 24GB RAM, RTX 4080 12GB)
- [x] Git repository initialized
- [x] Project structure planned

## Next Steps
1. [ ] Create GitHub private repo and push
2. [ ] Set up Docker Compose with all services
3. [ ] Create backend skeleton (FastAPI)
4. [ ] Set up Ollama with Mistral-7B
5. [ ] Create entity extractor module

## Decisions Made
- **LLM Strategy**: Local first (Ollama + Mistral-7B), abstracted interface for cloud fallback
- **Architecture**: Portable Docker Compose, all config via environment variables
- **Resource Limits**: ~3GB RAM when running, 0 when stopped

## Key Files
- `docker-compose.yml` - All services
- `.env` - Configuration (git ignored)
- `Makefile` - Common commands
- `backend/app/core/llm_provider.py` - Abstracted LLM interface

## Tech Stack
| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python 3.12 |
| Frontend | Next.js 14 |
| LLM (Local) | Ollama + Mistral-7B |
| LLM (Cloud) | Claude API (fallback) |
| Knowledge Graph | Neo4j |
| Database | PostgreSQL |
| Cache | Redis |

## How to Resume
1. Read this file
2. Check current phase
3. Continue from "Next Steps"
