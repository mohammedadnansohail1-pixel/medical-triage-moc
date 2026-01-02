# 🏥 Medical Triage AI System

An AI-powered medical triage system using a **multi-agent hierarchical architecture** with safety-first design. Routes patients to appropriate medical specialties based on symptom analysis.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 78.1% |
| **Weighted F1** | 77.6% |
| **Macro F1** | 66.1% |
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
                                   │  (7 classes, 99.9% on raw data) │
                                   └─────────────────────────────────┘
                                                    │
                                                    ▼
                                   ┌─────────────────────────────────┐
                                   │  Stage 5: Specialty Agent       │
                                   │  Generate differential diagnosis│
                                   │  within predicted specialty     │
                                   └─────────────────────────────────┘
                                                    │
                                                    ▼
                                   ┌─────────────────────────────────┐
                                   │  Stage 6: LLM Explanation       │
                                   │  Mistral 7B via Ollama          │
                                   │  Patient-friendly explanation   │
                                   └─────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Final Output                               │
│  {                                                                   │
│    "specialty": "cardiology",                                        │
│    "confidence": 0.94,                                               │
│    "differential_diagnosis": ["Angina", "MI", "Costochondritis"],   │
│    "explanation": "Your symptoms suggest a cardiac evaluation...",  │
│    "urgency": "high"                                                 │
│  }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔑 Key Features

### Safety-First Design
- **Emergency detection** runs BEFORE any ML model
- Rule-based keyword matching for life-threatening conditions
- Zero tolerance for missed emergencies (false negatives)

### Multi-Agent Architecture
- **Tier 1**: Specialty routing (XGBoost + SapBERT)
- **Tier 2**: Differential diagnosis (Specialty Agents)
- **Tier 3**: Patient explanation (LLM)

### Medical NLP Stack
- **SapBERT**: Biomedical entity linking (PubMedBERT-based)
- **XGBoost**: Fast, interpretable specialty classification
- **Mistral 7B**: Local LLM for explanations (via Ollama)

## 📁 Project Structure
```
medical-triage-moc/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI endpoints
│   │   ├── core/             # Core ML components
│   │   │   ├── triage_pipeline_v2.py    # Main orchestrator
│   │   │   ├── emergency_detector.py    # Safety rules
│   │   │   ├── symptom_normalizer.py    # Text preprocessing
│   │   │   ├── sapbert_linker.py        # Medical NER
│   │   │   ├── specialty_agent.py       # DDx generation
│   │   │   ├── explanation_generator.py # LLM integration
│   │   │   └── classifier/              # XGBoost training
│   │   ├── evaluation/       # Metrics framework
│   │   │   ├── metrics.py    # F1, precision, recall
│   │   │   ├── calibration.py # Brier, ECE
│   │   │   └── safety.py     # Emergency sensitivity
│   │   └── main.py           # FastAPI app
│   ├── data/
│   │   └── classifier/       # Trained models
│   ├── evaluation/           # Test cases
│   └── tests/                # Unit tests
├── data/
│   └── ddxplus/              # DDXPlus dataset
├── docker-compose.yml
├── Makefile
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (recommended)
- 16GB RAM minimum
- Ollama (for LLM explanations)

### Installation
```bash
# Clone repository
git clone https://github.com/mohammedadnansohail1-pixel/medical-triage-moc.git
cd medical-triage-moc

# Create virtual environment
python -m venv backend/venv
source backend/venv/bin/activate  # Linux/Mac
# or: backend\venv\Scripts\activate  # Windows

# Install dependencies
cd backend
pip install -r requirements.txt

# Download DDXPlus data (if not present)
# Place release_evidences.json and release_conditions.json in data/ddxplus/

# Install Ollama and pull Mistral (optional, for explanations)
ollama pull mistral
```

### Running the API
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

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

## 🧪 Evaluation

Run the full evaluation suite:
```bash
cd backend
python -c "
from app.evaluation.benchmark import run_benchmark
results = run_benchmark(n_samples=500)
results.print_summary()
"
```

### Metrics Computed
- **Classification**: Accuracy, Macro/Weighted F1, Per-class P/R/F1
- **Calibration**: Brier Score, ECE (Expected Calibration Error), MCE
- **Safety**: Emergency Sensitivity/Specificity, Under/Over-triage rates

## 📈 Training

### Retrain XGBoost Classifier
```bash
cd backend
python -m app.core.classifier.train
```

### Retrain on SapBERT-recovered codes (recommended)
```bash
cd backend
python scripts/retrain_xgboost_sapbert.py
```

## 🔬 Technical Details

### Dataset
- **DDXPlus**: Synthetic medical diagnosis dataset
- 49 conditions across 7 specialties
- 223 evidence codes (symptoms, risk factors)
- ~50,000 training samples

### Models
| Component | Model | Size |
|-----------|-------|------|
| Entity Linking | SapBERT (PubMedBERT) | 110M params |
| Classifier | XGBoost | ~1MB |
| Explanations | Mistral 7B | 7B params |

### Known Limitations
1. **DDXPlus data alignment**: Synthetic data doesn't perfectly match real symptom patterns
2. **Dermatology/GI accuracy**: Lower due to training data gaps
3. **Emergency class**: 0% DDXPlus sensitivity (rules catch real emergencies)

## 📄 Documentation

- [Accuracy Analysis](backend/ACCURACY_ANALYSIS.md) - Root cause analysis of performance gaps
- [Evaluation Summary](backend/EVALUATION_SUMMARY.md) - Detailed metrics breakdown
- [Phase 4 Research](PHASE4_RESEARCH.md) - Architecture decisions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This system is for educational and research purposes only.** It is NOT intended for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## 🙏 Acknowledgments

- [DDXPlus Dataset](https://github.com/mila-iqia/ddxplus) - Mila Quebec AI Institute
- [SapBERT](https://github.com/cambridgeltl/sapbert) - Cambridge LTL
- [Ollama](https://ollama.ai/) - Local LLM inference
