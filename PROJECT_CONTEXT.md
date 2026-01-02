# PROJECT CONTEXT - Medical Triage MOC

## Current Phase: Hybrid Ensemble Improvement (Step 3 of 5)

## Problem Statement
- Classifier trained on DDXPlus evidence codes: **100% accuracy**
- Full API with text input: **~50% accuracy**
- Root cause: Information loss when converting text → evidence codes

## Research Findings (5+ sources)

| Source | Approach | Accuracy | Works on Natural Text? |
|--------|----------|----------|------------------------|
| arXiv 2408.15827 | DDXPlus → text templates → transformer | 97% F1 | ✅ Yes |
| PMC5709846 | Clinical notes → subdomain classifier | 90%+ | ✅ Yes |
| PMC9862953 | Hybrid (BERT+LSTM+CNN+TF-IDF) | 93.5% | ✅ Yes |
| Nature 2024 | MCN-BERT on symptom descriptions | 99.58% | ✅ Yes |
| Our current | XGBoost on evidence codes | 100% | ❌ No |

**Key Insight from arXiv 2408.15827:**
> "short sentence templates that act as a response to each of the 223 unique questions... 
> written in first person... construct a block of text (patient report) that portrays 
> a patient reporting their own symptoms"

## Approved Plan

| Step | Task | Status | Time |
|------|------|--------|------|
| 1 | Parse release_evidences.json → build text templates | ⬜ Pending | 30 min |
| 2 | Convert DDXPlus evidence codes → natural language | ⬜ Pending | 30 min |
| 3 | Train text classifier (TF-IDF + XGBoost) | ⬜ Pending | 30 min |
| 4 | Update ensemble to use text classifier | ⬜ Pending | 15 min |
| 5 | Evaluate on DDXPlus via text API | ⬜ Pending | 15 min |

**Expected Result:** 80-90% accuracy on natural language input

## Completed Work

### Phase 1-2: Core System ✅
- FastAPI backend
- Entity Extractor (20 symptoms, negation, duration, severity)
- Knowledge Graph (Neo4j: 8 specialties, 20 symptoms, 19 diseases)
- LLM Router (llama3.1:8b)
- Rule-based emergency detection
- Ensemble combiner

### Phase 3: Evaluation ✅
- DDXPlus benchmark integration
- Curated test cases (22 cases, 86.4% accuracy)
- Evidence-code classifier (100% on DDXPlus)

### Hybrid Ensemble (Partial) ✅
- XGBoost classifier trained on 50k DDXPlus samples
- Classifier achieves 100% on evidence codes
- Ensemble weights: Classifier 40% + LLM 40% + KG 20%

## Current Evaluation Results

### Classifier-Only (Evidence Codes)
```
Accuracy: 100% (998/998)
- cardiology:        100% (137/137)
- dermatology:       100% (26/26)
- emergency:         100% (55/55)
- gastroenterology:  100% (74/74)
- general_medicine:  100% (297/297)
- neurology:         100% (97/97)
- pulmonology:       100% (312/312)
```

### Full API (Text Input) - BEFORE FIX
```
Accuracy: ~50%
Problem: Text → evidence code mapping loses information
```

## Architecture
```
Patient Text Input
       │
       ▼
┌──────────────────┐
│ Entity Extractor │
└────────┬─────────┘
         │
         ▼
   ┌─────┴─────┬────────────┐
   │           │            │
   ▼           ▼            ▼
┌──────┐  ┌────────┐  ┌──────────┐
│  KG  │  │  Text  │  │   LLM    │
│Router│  │Classif.│  │  Router  │
│(20%) │  │ (40%)  │  │  (40%)   │
└──┬───┘  └───┬────┘  └────┬─────┘
   │          │            │
   └────┬─────┴────────────┘
        │
        ▼
┌───────────────────┐
│ Weighted Ensemble │
└────────┬──────────┘
         │
         ▼
    Final Specialty
```

## Key Files

### Classifier (Current - Evidence-based)
- `backend/app/core/classifier/prepare_data.py` - Data preparation
- `backend/app/core/classifier/train.py` - XGBoost training
- `backend/app/core/classifier/router.py` - Routing module
- `data/classifier/model.pkl` - Trained model
- `data/classifier/vocabulary.pkl` - Evidence code vocab

### Evaluation
- `backend/evaluation/ddxplus_eval.py` - DDXPlus benchmark
- `backend/evaluation/runner.py` - Curated test runner
- `backend/evaluation/test_cases.py` - 22 curated cases

### Core System
- `backend/app/core/ensemble.py` - Ensemble router
- `backend/app/core/entity_extractor.py` - Symptom extraction
- `backend/app/core/knowledge_graph.py` - Neo4j router
- `backend/app/core/llm_router.py` - Ollama LLM router

## How to Run
```bash
# Start services
cd ~/projects/medical-triage-moc
docker compose up -d neo4j ollama

# Start backend
cd backend && source venv/bin/activate
PYTHONPATH=. uvicorn app.main:app --reload

# Run evaluation
PYTHONPATH=. python -m evaluation.ddxplus_eval
```

## Next Steps
1. Build text templates from release_evidences.json
2. Generate natural language training data from DDXPlus
3. Train TF-IDF + XGBoost on text
4. Integrate text classifier into ensemble
5. Re-evaluate on DDXPlus

## Git Repository
https://github.com/mohammedadnansohail1-pixel/medical-triage-moc
