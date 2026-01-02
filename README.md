# Medical Triage AI System

An intelligent medical symptom triage system using a hybrid ML pipeline for specialty routing and differential diagnosis generation.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Tests](https://img.shields.io/badge/tests-51%20passed-success)
![Accuracy](https://img.shields.io/badge/accuracy-99.9%25-brightgreen)

## Overview

This system processes patient symptoms and demographics to:
1. **Route to appropriate medical specialty** (7 specialties)
2. **Generate differential diagnosis** (ranked by probability)
3. **Provide LLM-powered explanations** with urgency levels

### Key Metrics

| Metric | Value |
|--------|-------|
| DDXPlus Accuracy | 99.90% |
| Natural Language Accuracy | 100% |
| Emergency Detection | 100% |
| API Response Time | <100ms |
| Test Coverage | 51 tests |

## Architecture
```
Patient Input (symptoms, age, sex)
              ↓
┌─────────────────────────────────────────┐
│  1. Symptom Normalization (rule-based)  │
│  2. Emergency Detection (rule-based)    │  ← 100% reliable
│  3. Specialty Rules (keyword override)  │
│  4. SapBERT Entity Linking (CUDA)       │  ← Medical NER
│  5. XGBoost Classification (7-class)    │  ← 99.9% accuracy
│  6. Bayesian Differential Diagnosis     │
│  7. LLM Explanation (llama3.1:8b)       │
└─────────────────────────────────────────┘
              ↓
Response: specialty, confidence, DDx[], explanation
```

## Supported Specialties

| Specialty | Detection Method | Confidence |
|-----------|------------------|------------|
| Emergency | Rule-based | 100% |
| Cardiology | ML Classification | 99%+ |
| Pulmonology | ML Classification | 99%+ |
| Neurology | ML Classification | 99%+ |
| Gastroenterology | ML + Rules | 80-99% |
| Dermatology | Rule-based | 85% |
| General Medicine | ML Classification | 99%+ |

## Quick Start

### Prerequisites
- Python 3.12+
- Docker (optional)
- Ollama with llama3.1:8b (for explanations)

### Local Development
```bash
# Clone and setup
git clone <repo-url>
cd medical-triage-moc/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --port 8000
```

### Docker Deployment
```bash
# CPU only (no LLM explanations)
docker-compose -f docker-compose.cpu.yml up -d

# With GPU + Ollama (full features)
docker-compose up -d
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Triage Request
```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["chest pain", "shortness of breath"],
    "age": 55,
    "sex": "male",
    "include_explanation": true
  }'
```

### Response Example
```json
{
  "specialty": "emergency",
  "confidence": 1.0,
  "differential_diagnosis": [],
  "explanation": {
    "text": "These symptoms require immediate emergency evaluation...",
    "urgency": "emergency",
    "next_steps": ["Call 911", "Do not drive yourself"]
  },
  "route": "EMERGENCY_OVERRIDE"
}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root info |
| GET | `/docs` | Swagger UI |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/triage` | Process symptoms |

## Project Structure
```
medical-triage-moc/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI application
│   │   ├── api/triage.py                # Triage endpoint
│   │   └── core/
│   │       ├── triage_pipeline_v2.py    # Main orchestrator
│   │       ├── emergency_detector.py    # Rule-based safety
│   │       ├── symptom_normalizer.py    # Text preprocessing
│   │       ├── sapbert_linker.py        # Medical NER (SapBERT)
│   │       ├── specialty_agent.py       # DDx generation
│   │       └── explanation_generator.py # LLM integration
│   ├── tests/
│   │   └── test_api_comprehensive.py    # 51 API tests
│   ├── data/classifier/
│   │   ├── model.pkl                    # XGBoost model
│   │   └── vocabulary.pkl               # Feature vocabulary
│   ├── Dockerfile
│   └── requirements.txt
├── data/ddxplus/
│   ├── release_evidences.json           # Symptom definitions
│   └── condition_model.json             # Disease-symptom priors
├── docs/
│   ├── TECHNICAL_ARCHITECTURE.md
│   └── METRICS_ANALYSIS.md
├── docker-compose.yml                   # GPU + Ollama
├── docker-compose.cpu.yml               # CPU only
└── PROJECT_CONTEXT.md
```

## Testing
```bash
cd backend

# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_api_comprehensive.py::TestEmergencyDetection -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Test Categories
- Health Endpoints (3 tests)
- Input Validation (10 tests)
- Specialty Routing (11 tests)
- Emergency Detection (10 tests)
- Response Structure (4 tests)
- Confidence Scores (3 tests)
- Demographics (3 tests)
- Performance (2 tests)
- Security Inputs (4 tests)

## Route Types

| Route | Description | Confidence |
|-------|-------------|------------|
| `EMERGENCY_OVERRIDE` | Rule-based emergency detection | 100% |
| `RULE_OVERRIDE` | Keyword-based specialty match | 80-85% |
| `ML_CLASSIFICATION` | XGBoost prediction | Variable |
| `DEFAULT_FALLBACK` | No codes matched | 50% |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LLM` | `true` | Enable LLM explanations |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `LOG_LEVEL` | `INFO` | Logging level |

## Documentation

- [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md) - System design details
- [Metrics Analysis](docs/METRICS_ANALYSIS.md) - Evaluation results
- [Project Context](PROJECT_CONTEXT.md) - Development status

## Known Limitations

1. **DDXPlus Dataset**: Limited to 49 conditions (synthetic data)
2. **Dermatology**: Rule-based only (no DDXPlus training data)
3. **Language**: English only
4. **Images**: No image/photo analysis

## Future Roadmap

- [ ] Frontend UI (React/Next.js)
- [ ] Multi-turn conversation
- [ ] Real clinical data training
- [ ] Multi-language support
- [ ] Image analysis integration

## License

MIT License - See LICENSE file

## Author

Mohammed Adnan Sohail

---

**⚠️ Disclaimer**: This system is for educational/research purposes only. Not intended for actual medical diagnosis. Always consult healthcare professionals for medical advice.
