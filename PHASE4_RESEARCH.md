# Phase 4 Research: Specialized Agent Routing

## Research Summary

### 1. Multi-Agent Architecture Patterns (from arxiv, industry)

**Three-Agent Pattern (Academic - arxiv):**
- RecipientAgent: Transforms unstructured symptoms → standardized HPI
- InquirerAgent: Identifies missing info, asks targeted questions
- DepartmentAgent: Makes department recommendations

**Orchestrator-Specialist-Observer Pattern (Industry - TATEEDA):**
- Orchestrator: Breaks request into steps, routes work
- Specialist Agent: Does one job well (focused domain)
- Observer/Verifier: Validates outputs against rules before delivery

**Key Insight:** Production systems use iterative multi-round interactions, not single-shot routing.

---

### 2. Confidence-Based Routing Strategies

**Threshold-Based Routing:**
- High confidence (>80%): Route directly to specialty agent
- Medium confidence (50-80%): Ask clarifying questions first
- Low confidence (<50%): Route to general_medicine or human escalation

**Uncertainty-Based Fallback:**
- If model uncertain → defer to stronger model or human
- If confidence gap between top-2 predictions is small → ask clarifying questions

**Coverage vs Accuracy Trade-off:**
- Higher threshold = better accuracy, lower coverage
- Sweet spot typically around 0.6-0.8 threshold

---

### 3. Safety & Guardrails (Critical for Medical)

**Must-Have Guardrails:**
1. **Medical Disclaimer**: Never diagnose, only triage
2. **Emergency Escalation**: Always override to emergency for red flags
3. **Hallucination Prevention**: Don't invent symptoms or conditions
4. **Human Escalation Path**: Clear exit to real healthcare
5. **Audit Logging**: Track all interactions for review

**Input Guardrails:**
- Block prompt injection attempts
- Detect off-topic queries
- Identify mental health crisis signals

**Output Guardrails:**
- No specific drug dosages
- No definitive diagnoses
- Always recommend professional consultation
- Detect harmful advice patterns

---

### 4. Failure Scenarios & Mitigations

| Failure | Impact | Mitigation |
|---------|--------|------------|
| LLM unavailable | Service down | Fallback to rule-based response |
| Wrong specialty routed | Poor advice | Confidence threshold + clarifying questions |
| Agent gives harmful advice | Patient harm | Output guardrails + disclaimers |
| Emergency missed | Life risk | Rule-based emergency detection BEFORE ML |
| Patient in crisis | Safety risk | Crisis detection → immediate resources |
| Hallucination | Misinformation | Ground responses in evidence codes |

---

### 5. Architecture Decision

**Chosen: Orchestrator + Specialized Agents + Guardrails**
```
User Input
    ↓
[Input Guardrails] ─── Block harmful/off-topic
    ↓
[Triage Pipeline] ─── 70% accuracy, confidence score
    ↓
[Routing Decision]
    ├── confidence >= 0.7 → Direct to Specialty Agent
    ├── confidence 0.4-0.7 → Clarifying Questions first
    └── confidence < 0.4 → General Medicine + disclaimer
    ↓
[Specialty Agent] ─── Domain-specific prompts
    ↓
[Output Guardrails] ─── No diagnosis, add disclaimers
    ↓
User Response
```

---

### 6. Implementation Plan

**Phase 4A: Core Routing (MVP)**
1. Routing logic based on confidence thresholds
2. 7 specialized agent prompts
3. Basic conversation state

**Phase 4B: Guardrails**
1. Input validation (off-topic, injection)
2. Output validation (no diagnosis, disclaimers)
3. Emergency override at all stages

**Phase 4C: Conversation Flow**
1. Multi-turn conversations
2. Clarifying questions for low confidence
3. Session state management

**Phase 4D: Error Handling**
1. LLM fallback responses
2. Graceful degradation
3. Audit logging

---

### 7. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single vs Multiple LLMs | Single (llama3.1:8b) | VRAM constraint |
| Stateful vs Stateless | Stateful (per-session) | Need conversation context |
| Sync vs Async | Async | LLM calls are slow |
| Confidence threshold | 0.7 (high), 0.4 (low) | Balance accuracy/coverage |
| Emergency handling | Rule-based BEFORE ML | 100% safety requirement |

---

### 8. API Design
```
POST /api/chat
{
  "session_id": "uuid",
  "message": "I have chest pain and difficulty breathing"
}

Response:
{
  "session_id": "uuid",
  "response": "I understand you're experiencing chest pain...",
  "specialty": "cardiology",
  "confidence": 0.85,
  "is_emergency": false,
  "requires_clarification": false,
  "disclaimer": "This is not medical advice..."
}
```

---

### 9. Testing Strategy

1. **Unit Tests**: Each agent prompt produces expected format
2. **Integration Tests**: Full flow from input to output
3. **Safety Tests**: Guardrails block harmful content
4. **Failure Tests**: Graceful handling when LLM fails
5. **Edge Cases**: Ambiguous symptoms, multi-specialty

---

### 10. Success Metrics

- Routing accuracy: >70% correct specialty
- Emergency detection: 100% recall
- Response time: <5s for non-complex
- User safety: 0 harmful advice
- Graceful failures: 100% handled
