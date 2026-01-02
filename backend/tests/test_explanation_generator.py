"""
Tests for LLM Explanation Generator.

Run: pytest tests/test_explanation_generator.py -v
"""

import pytest
from unittest.mock import Mock, patch

from app.core.explanation_generator import (
    ExplanationGenerator,
    ExplanationResult,
    get_explanation_generator,
)


class TestExplanationResult:
    """Test ExplanationResult dataclass."""

    def test_create_result(self):
        result = ExplanationResult(
            explanation="Test explanation",
            urgency="urgent",
            next_steps=["Step 1", "Step 2"],
        )
        assert result.explanation == "Test explanation"
        assert result.urgency == "urgent"
        assert len(result.next_steps) == 2


class TestExplanationGenerator:
    """Test ExplanationGenerator."""

    @pytest.fixture
    def generator(self):
        return ExplanationGenerator()

    def test_init_defaults(self, generator):
        assert generator.model == "llama3.1:8b"
        assert generator.base_url == "http://localhost:11434"

    def test_build_prompt(self, generator):
        prompt = generator._build_prompt(
            symptoms=["cough", "fever"],
            specialty="pulmonology",
            confidence=0.95,
            differential_diagnosis=[{"condition": "Bronchitis", "probability": 0.9}],
            age=45,
            sex="male",
        )
        assert "cough, fever" in prompt
        assert "pulmonology" in prompt
        assert "Bronchitis" in prompt
        assert "45yo" in prompt
        assert "male" in prompt

    def test_fallback_explanation_emergency(self, generator):
        result = generator._fallback_explanation("emergency", [])
        assert result.urgency == "emergency"
        assert "911" in result.next_steps[0]

    def test_fallback_explanation_cardiology(self, generator):
        result = generator._fallback_explanation(
            "cardiology",
            [{"condition": "Unstable angina", "probability": 0.8}]
        )
        assert result.urgency == "urgent"
        assert "Unstable angina" in result.explanation
        assert "cardiologist" in result.next_steps[0].lower()

    def test_fallback_explanation_general(self, generator):
        result = generator._fallback_explanation("general_medicine", [])
        assert result.urgency == "routine"
        assert "General Medicine" in result.explanation

    def test_parse_valid_json(self, generator):
        raw = '{"explanation": "Test", "urgency": "urgent", "next_steps": ["Step 1"]}'
        result = generator._parse_response(raw, "cardiology", [])
        assert result.explanation == "Test"
        assert result.urgency == "urgent"
        assert result.next_steps == ["Step 1"]

    def test_parse_invalid_json_uses_fallback(self, generator):
        raw = "This is not JSON"
        result = generator._parse_response(raw, "pulmonology", [])
        # Should use fallback
        assert "Pulmonology" in result.explanation
        assert result.urgency == "urgent"


@pytest.mark.skipif(
    not ExplanationGenerator().is_available(),
    reason="Ollama not running"
)
class TestExplanationGeneratorIntegration:
    """Integration tests (require Ollama running)."""

    @pytest.fixture
    def generator(self):
        return ExplanationGenerator()

    def test_generate_respiratory(self, generator):
        result = generator.generate(
            symptoms=["cough", "fever"],
            specialty="pulmonology",
            confidence=0.95,
            differential_diagnosis=[{"condition": "Bronchitis", "probability": 0.9}],
            age=40,
            sex="male",
        )
        assert result.explanation
        assert result.urgency in ["emergency", "urgent", "routine"]
        assert len(result.next_steps) > 0

    def test_generate_cardiac(self, generator):
        result = generator.generate(
            symptoms=["chest pain", "palpitations"],
            specialty="cardiology",
            confidence=0.99,
            differential_diagnosis=[{"condition": "Atrial fibrillation", "probability": 0.95}],
            age=60,
            sex="male",
        )
        assert result.explanation
        assert result.urgency in ["emergency", "urgent"]

    def test_is_available(self, generator):
        assert generator.is_available() == True


class TestSingleton:
    """Test singleton pattern."""

    def test_get_generator_returns_same_instance(self):
        gen1 = get_explanation_generator()
        gen2 = get_explanation_generator()
        assert gen1 is gen2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
