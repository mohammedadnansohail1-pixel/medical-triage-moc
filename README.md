# 🏥 Medical Triage AI System

An AI-powered medical triage system using a **multi-agent hierarchical architecture** with safety-first design. Routes patients to appropriate medical specialties based on symptom analysis.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 78.1% |
| **Weighted F1** | 77.6% |
| **Macro F1** | 66.1% |
| **Model Training Accuracy** | 99.9% |
| **Avg Latency** | 8.7ms |

### Per-Specialty Performance

| Specialty | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Cardiology | 86.8% | 96.5% | **91.4%** |
| Neurology | 91.8% | 82.7% | **87.0%** |
| Pulmonology | 80.6% | 85.5% | **83.0%** |
| General Medicine | 81.2% | 77.8% | **79.5%** |
| Gastroenterology | 60.8% | 59.3% | 60.0% |
| Dermatology | 66.7% | 57.1% | 61.5% |

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- GPU optional (CUDA for faster inference)

### Option 1: Using Make (Recommended)
```bash
# Clone the repository
git clone https://github.com/mohammedadnansohail1-pixel/medical-triage-moc.git
cd medical-triage-moc

# Complete setup (creates venv, installs deps, trains model)
make setup

# Start the API server
make run
```

### Option 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/mohammedadnansohail1-pixel/medical-triage-moc.git
cd medical-triage-moc

# Create virtual environment
python3 -m venv backend/venv

# Activate virtual environment
source backend/venv/bin/activate  # Linux/Mac
# OR
backend\venv\Scripts\activate     # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Train the models (required first time)
cd backend
python -m scripts.train_models

# Verify installation
python -m scripts.verify_install

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Using Docker
```bash
docker-compose up -d
```

---

## 🧪 Test the API

Once the server is running, open http://localhost:8000/docs for interactive API docs.

### Example Request
```bash
curl -X POST "http://localhost:8000/api/triage" \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["chest pain", "shortness of breath", "sweating"],
    "age": 55,
    "sex": "male"
  }'
```

### Example Response
```json
{
  "specialty": "emergency",
  "confidence": 1.0,
  "route": "EMERGENCY_OVERRIDE",
  "reasoning": ["Emergency detected: cardiac_emergency"],
  "differential_diagnosis": [],
  "explanation": {
    "text": "Your symptoms indicate a medical emergency...",
    "urgency": "emergency",
    "next_steps": ["Call 911 immediately", "Go to nearest emergency room"]
  }
}
```

### More Test Cases
```bash
# Pulmonology case
curl -X POST "http://localhost:8000/api/triage" \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["cough", "fever", "difficulty breathing"]}'

# Gastroenterology case
curl -X POST "http://localhost:8000/api/triage" \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["stomach pain", "nausea", "vomiting"]}'

# Neurology case  
curl -X POST "http://localhost:8000/api/triage" \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["severe headache", "dizziness", "numbness"]}'
```

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                         Patient Input                                │
│                    "chest pain, shortness of breath"                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Stage 1: Symptom Normalization                     │
│              Patient language → Medical terminology                  │
│         "tummy ache" → "abdominal pain gastric discomfort"          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Stage 2: Emergency Detection (Rule-Based)            │
│                      🚨 SAFETY-CRITICAL LAYER 🚨                     │
│    Keywords: chest pain + SOB, stroke symptoms, severe bleeding     │
│                     100% Reliable - No ML Failures                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              Emergency?                      Not Emergency
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐      ┌─────────────────────────────────┐
        │  EMERGENCY_OVERRIDE│      │  Stage 3: SapBERT Linking       │
        │  → 911 / ER        │      │  Natural language → Evidence    │
        └───────────────────┘      │  codes (DDXPlus E_xxx format)   │
                                   └─────────────────────────────────┘
                                                    │
                                                    ▼
                                   ┌─────────────────────────────────┐
                                   │  Stage 4: XGBoost Classifier    │
                                   │  Evidence codes → Specialty     │
                                   │  (7 classes, 99.9% accuracy)    │
                                   └─────────────────────────────────┘
                                                    │
                                                    ▼
                                   ┌─────────────────────────────────┐
                                   │  Stage 5: Specialty Agent       │
                                   │  Generate differential diagnosis│
                                   └─────────────────────────────────┘
                                                    │
                                                    ▼
                                   ┌─────────────────────────────────┐
                                   │  Stage 6: LLM Explanation       │
                                   │  (Optional - requires Ollama)   │
                                   └─────────────────────────────────┘
```

---

## 📁 Project Structure
```
medical-triage-moc/
├── backend/
│   ├── app/
│   │   ├── api/                    # FastAPI endpoints
│   │   │   └── triage.py           # /api/triage endpoint
│   │   ├── core/                   # Core ML components
│   │   │   ├── triage_pipeline_v2.py    # Main orchestrator
│   │   │   ├── emergency_detector.py    # Safety rules
│   │   │   ├── symptom_normalizer.py    # Text preprocessing
│   │   │   ├── sapbert_linker.py        # Medical NER
│   │   │   ├── specialty_agent.py       # DDx generation
│   │   │   └── explanation_generator.py # LLM integration
│   │   ├── evaluation/             # Metrics framework
│   │   └── main.py                 # FastAPI app
│   ├── scripts/
│   │   ├── train_models.py         # Model training script
│   │   └── verify_install.py       # Installation checker
│   ├── data/classifier/            # Trained models (generated)
│   └── requirements.txt
├── data/
│   └── ddxplus/                    # DDXPlus dataset files
├── scripts/
│   └── setup.sh                    # Setup script
├── Makefile                        # Easy commands
├── docker-compose.yml
└── README.md
```

---

## 🔧 Available Commands
```bash
make help      # Show all commands
make setup     # First-time setup (venv + deps + train)
make install   # Install dependencies only
make train     # Train models from scratch
make run       # Start API server
make test      # Run tests
make evaluate  # Run evaluation metrics
make clean     # Remove generated files
```

---

## 🔬 Technical Details

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Entity Linking | SapBERT (PubMedBERT) | Convert symptoms to medical codes |
| Classifier | XGBoost | Route to medical specialty |
| Explanations | Mistral 7B (optional) | Patient-friendly explanations |

### Dataset

- **DDXPlus**: Synthetic medical diagnosis dataset from Mila Quebec AI Institute
- 49 conditions across 7 specialties
- 223 evidence codes (symptoms, risk factors)

### Specialties Supported

1. **Emergency** - Life-threatening conditions
2. **Cardiology** - Heart-related issues
3. **Pulmonology** - Respiratory conditions
4. **Neurology** - Neurological disorders
5. **Gastroenterology** - Digestive system
6. **Dermatology** - Skin conditions
7. **General Medicine** - Other conditions

---

## 📈 Training Your Own Model
```bash
# Retrain with default settings
make train

# Or manually with custom settings
cd backend
python -m scripts.train_models
```

The training script will:
1. Load DDXPlus symptom probabilities
2. Generate synthetic training data
3. Train XGBoost classifier
4. Save model to `backend/data/classifier/`

---

## 🔍 Troubleshooting

### "Module not found" errors
```bash
# Make sure virtual environment is activated
source backend/venv/bin/activate
pip install -r backend/requirements.txt
```

### "Model file not found" errors
```bash
# Train the models first
make train
# Or
cd backend && python -m scripts.train_models
```

### CUDA/GPU issues
```bash
# Install CPU-only PyTorch if you don't have a GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Verify installation
```bash
cd backend && python -m scripts.verify_install
```

---

## 📄 Documentation

- [Accuracy Analysis](backend/ACCURACY_ANALYSIS.md) - Root cause analysis
- [Evaluation Summary](backend/EVALUATION_SUMMARY.md) - Detailed metrics

---

## ⚠️ Disclaimer

**This system is for educational and research purposes only.** It is NOT intended for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [DDXPlus Dataset](https://github.com/mila-iqia/ddxplus) - Mila Quebec AI Institute
- [SapBERT](https://github.com/cambridgeltl/sapbert) - Cambridge LTL
- [Ollama](https://ollama.ai/) - Local LLM inference
