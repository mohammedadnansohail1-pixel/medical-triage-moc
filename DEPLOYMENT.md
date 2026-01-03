# Deployment Guide

Complete guide for deploying the Medical Triage AI system.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Manual Installation](#manual-installation)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Production Checklist](#production-checklist)

---

## Quick Start

### Option A: Docker (Recommended)
```bash
# GPU (NVIDIA)
docker-compose up -d

# CPU only
docker-compose -f docker-compose.cpu.yml up -d
```

### Option B: Manual
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Docker Deployment

### Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker-compose --version` |
| NVIDIA Driver | 525+ | `nvidia-smi` |
| nvidia-docker | 2.0+ | `nvidia-container-cli --version` |

### GPU Deployment (Recommended)
```bash
# Clone repository
git clone https://github.com/yourusername/medical-triage-moc.git
cd medical-triage-moc

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

**Services Started:**
- `api` (port 8000) - Main triage API
- `ollama` (port 11434) - LLM server
- `ollama-init` - Downloads models (runs once)

### CPU-Only Deployment

For machines without NVIDIA GPU:
```bash
docker-compose -f docker-compose.cpu.yml up -d
```

**Note:** CPU inference is 5-10x slower than GPU.

### Verify Deployment
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Conversation health
curl http://localhost:8000/api/v1/conversation/health

# Test triage
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["headache", "fever"], "age": 30, "sex": "male"}'

# Test conversation
curl -X POST http://localhost:8000/api/v1/conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a headache"}'
```

### Stop Services
```bash
docker-compose down

# Remove volumes (deletes downloaded models)
docker-compose down -v
```

---

## Manual Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Ollama | Latest |
| CUDA (optional) | 11.8+ |

### Step 1: Install Ollama
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# Download from https://ollama.com/download
```

### Step 2: Start Ollama & Download Model
```bash
# Start Ollama service
ollama serve &

# Download Llama 3.1 8B
ollama pull llama3.1:8b

# Verify
ollama list
```

### Step 3: Setup Python Environment
```bash
cd medical-triage-moc/backend

# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run Application
```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 5: Run Tests
```bash
# All tests
pytest tests/ -v

# Conversation tests only
pytest tests/test_conversation.py -v

# Evaluation metrics
python -m app.evaluation.conversation_metrics
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `LOG_LEVEL` | `INFO` | Logging level |
| `PYTHONPATH` | `/app` | Python path |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache |
| `CUDA_VISIBLE_DEVICES` | (auto) | GPU selection |

### Setting Variables

**Docker:**
```yaml
# docker-compose.yml
environment:
  - OLLAMA_URL=http://ollama:11434
  - LOG_LEVEL=DEBUG
```

**Manual:**
```bash
export OLLAMA_URL=http://localhost:11434
export LOG_LEVEL=DEBUG
```

### Resource Requirements

| Deployment | RAM | VRAM | Storage | CPU |
|------------|-----|------|---------|-----|
| Minimum (CPU) | 16GB | - | 30GB | 4 cores |
| Recommended (GPU) | 16GB | 8GB | 50GB | 8 cores |
| Production (GPU) | 32GB | 16GB | 100GB | 16 cores |

---

## Verification

### Health Checks
```bash
# API health
curl http://localhost:8000/api/v1/health

# Expected response:
{
  "status": "healthy",
  "specialty_classifier": "loaded",
  "sapbert_linker": "loaded"
}
```
```bash
# Conversation health
curl http://localhost:8000/api/v1/conversation/health

# Expected response:
{
  "status": "healthy",
  "graph_nodes": ["emergency", "supervisor", "triage", "dermatology", "cardiology"],
  "agents": ["supervisor", "emergency", "triage", "dermatology", "cardiology"]
}
```

### Functional Tests

**Test 1: Basic Triage**
```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["cough", "fever", "sore throat"],
    "age": 35,
    "sex": "female"
  }'
```

**Test 2: Emergency Detection**
```bash
curl -X POST http://localhost:8000/api/v1/conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "I have severe chest pain and cannot breathe"}'

# Should return risk_level: "emergency"
```

**Test 3: Multi-turn Conversation**
```bash
# Turn 1
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a rash"}')

SESSION_ID=$(echo $RESPONSE | jq -r '.session_id')

# Turn 2
curl -X POST http://localhost:8000/api/v1/conversation \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION_ID\", \"message\": \"It started yesterday\"}"
```

### Run Full Evaluation
```bash
cd backend
python -m app.evaluation.conversation_metrics
```

Expected output:
```
📊 Emergency Detection Sensitivity: 100.0% (8/8)
📊 Routing Accuracy: 100.0% (12/12)
📊 Non-Emergency Specificity: 100.0% (5/5)
📊 Overall Accuracy: 100.0% (25/25)
✅ SAFETY CHECK PASSED: 100% emergency detection
```

---

## Production Checklist

### Security

- [ ] Enable HTTPS (use reverse proxy like nginx)
- [ ] Set up authentication/API keys
- [ ] Configure CORS for your domains
- [ ] Review and restrict network access
- [ ] Enable rate limiting

### Monitoring

- [ ] Set up health check alerts
- [ ] Configure log aggregation
- [ ] Monitor GPU/CPU usage
- [ ] Track response latencies
- [ ] Alert on emergency detections

### Backup

- [ ] Backup Ollama models volume
- [ ] Backup HuggingFace cache
- [ ] Document recovery procedures

### Scaling

- [ ] Use multiple API workers (`--workers 4`)
- [ ] Consider load balancer for high traffic
- [ ] Separate Ollama to dedicated GPU server

### Legal/Compliance

- [ ] Add medical disclaimer to UI
- [ ] Ensure HIPAA compliance if handling PHI
- [ ] Document data retention policies
- [ ] Get legal review for your jurisdiction

---

## Troubleshooting

### Docker Issues

**Container won't start:**
```bash
docker-compose logs api
docker-compose logs ollama
```

**GPU not detected:**
```bash
# Check nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

**Out of memory:**
- Reduce model size: Use `llama3.2:3b` instead of `llama3.1:8b`
- Increase Docker memory limit

### Manual Installation Issues

**Ollama connection refused:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve &
```

**CUDA not available:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be >= 1
```

**Model download fails:**
```bash
# Clear HuggingFace cache and retry
rm -rf ~/.cache/huggingface
python -c "from transformers import AutoModel; AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')"
```

---

## Support

- **Issues:** https://github.com/yourusername/medical-triage-moc/issues
- **Docs:** See `/docs` folder
- **API Docs:** http://localhost:8000/docs (Swagger UI)

---

*Last Updated: January 2, 2026*
