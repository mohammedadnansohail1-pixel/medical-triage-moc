# API Reference

## Medical Triage API v1

Base URL: `http://localhost:8000`

---

## Endpoints

### Health Check
```
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "pipeline_loaded": true
}
```

---

### Triage Symptoms
```
POST /api/v1/triage
```

Process patient symptoms and return specialty routing with differential diagnosis.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| symptoms | string[] | Yes | List of symptom descriptions |
| age | integer | No | Patient age (0-120) |
| sex | string | No | "male" or "female" |
| include_explanation | boolean | No | Generate LLM explanation (default: true) |

**Example Request:**
```json
{
  "symptoms": ["chest pain", "shortness of breath"],
  "age": 55,
  "sex": "male",
  "include_explanation": true
}
```

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| specialty | string | Recommended medical specialty |
| confidence | float | Confidence score (0.0-1.0) |
| differential_diagnosis | array | Ranked list of possible conditions |
| explanation | object | LLM-generated explanation (if requested) |
| route | string | Routing method used |

**Example Response:**
```json
{
  "specialty": "cardiology",
  "confidence": 0.812,
  "differential_diagnosis": [
    {
      "condition": "Unstable angina",
      "probability": 0.45,
      "rank": 1
    },
    {
      "condition": "PSVT",
      "probability": 0.32,
      "rank": 2
    },
    {
      "condition": "Atrial fibrillation",
      "probability": 0.15,
      "rank": 3
    }
  ],
  "explanation": {
    "text": "Based on your symptoms of chest pain and shortness of breath, cardiology evaluation is recommended...",
    "urgency": "urgent",
    "next_steps": [
      "Seek medical attention within 24 hours",
      "Avoid strenuous activity",
      "Call 911 if symptoms worsen"
    ]
  },
  "route": "ML_CLASSIFICATION"
}
```

---

## Route Types

| Route | Description | Confidence |
|-------|-------------|------------|
| EMERGENCY_OVERRIDE | Life-threatening symptoms detected | 100% |
| RULE_OVERRIDE | Keyword-based specialty match | 80-85% |
| ML_CLASSIFICATION | XGBoost model prediction | Variable |
| DEFAULT_FALLBACK | No evidence codes matched | 50% |

---

## Specialties

| Specialty | Description |
|-----------|-------------|
| emergency | Life-threatening conditions |
| cardiology | Heart-related conditions |
| pulmonology | Lung/respiratory conditions |
| neurology | Brain/nervous system conditions |
| gastroenterology | Digestive system conditions |
| dermatology | Skin conditions |
| general_medicine | General/unspecified conditions |

---

## Urgency Levels

| Level | Description | Action |
|-------|-------------|--------|
| emergency | Life-threatening | Call 911 immediately |
| urgent | Needs prompt attention | Seek care within 24 hours |
| routine | Non-urgent | Schedule appointment |

---

## Error Responses

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "symptoms"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Pipeline processing error: ..."
}
```

---

## Examples

### Emergency Case
```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["severe chest pain", "can not breathe", "sweating"],
    "include_explanation": false
  }'
```

Response:
```json
{
  "specialty": "emergency",
  "confidence": 1.0,
  "differential_diagnosis": [],
  "explanation": null,
  "route": "EMERGENCY_OVERRIDE"
}
```

### Routine Case
```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["runny nose", "mild cough"],
    "age": 25,
    "include_explanation": false
  }'
```

Response:
```json
{
  "specialty": "pulmonology",
  "confidence": 0.95,
  "differential_diagnosis": [
    {"condition": "URTI", "probability": 0.78, "rank": 1},
    {"condition": "Viral pharyngitis", "probability": 0.15, "rank": 2}
  ],
  "explanation": null,
  "route": "ML_CLASSIFICATION"
}
```

---

## Rate Limits

No rate limits in current version.

## Authentication

No authentication required in current version.

---

*API Version 1.0 - January 2026*
