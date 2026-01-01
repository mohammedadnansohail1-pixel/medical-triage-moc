"""Tests for entity extraction."""

import pytest
from app.core.entity_extractor import extract_entities, EntityExtractor


class TestSymptomExtraction:
    """Test symptom extraction."""

    def test_extract_chest_pain(self) -> None:
        """Test chest pain extraction."""
        result = extract_entities("I have severe chest pain")
        assert "chest_pain" in result.symptoms

    def test_extract_shortness_of_breath(self) -> None:
        """Test shortness of breath variations."""
        texts = [
            "I have shortness of breath",
            "difficulty breathing",
            "I can't breathe",
            "breathing problems",
        ]
        for text in texts:
            result = extract_entities(text)
            assert "shortness_of_breath" in result.symptoms, f"Failed for: {text}"

    def test_extract_multiple_symptoms(self) -> None:
        """Test extracting multiple symptoms."""
        result = extract_entities(
            "I have chest pain, shortness of breath, and I'm sweating"
        )
        assert "chest_pain" in result.symptoms
        assert "shortness_of_breath" in result.symptoms
        assert "sweating" in result.symptoms

    def test_extract_headache_variations(self) -> None:
        """Test headache pattern variations."""
        texts = ["bad headache", "pain in my head", "throbbing head"]
        for text in texts:
            result = extract_entities(text)
            assert "headache" in result.symptoms, f"Failed for: {text}"


class TestNegation:
    """Test negation detection."""

    def test_negated_symptom(self) -> None:
        """Test that negated symptoms are captured separately."""
        result = extract_entities("I have chest pain but no nausea")
        assert "chest_pain" in result.symptoms
        assert "nausea" in result.negated_symptoms
        assert "nausea" not in result.symptoms

    def test_denies_pattern(self) -> None:
        """Test 'denies' negation pattern."""
        result = extract_entities("Patient denies fever or vomiting")
        assert "fever" in result.negated_symptoms
        assert "vomiting" in result.negated_symptoms


class TestDuration:
    """Test duration extraction."""

    def test_hours_duration(self) -> None:
        """Test hour-based duration."""
        result = extract_entities("chest pain for 2 hours")
        assert result.duration == "2_hours"

    def test_days_duration(self) -> None:
        """Test day-based duration."""
        result = extract_entities("headache for 3 days")
        assert result.duration == "3_days"

    def test_sudden_onset(self) -> None:
        """Test sudden onset detection."""
        result = extract_entities("sudden chest pain")
        assert result.duration == "sudden_onset"

    def test_since_yesterday(self) -> None:
        """Test 'since yesterday' pattern."""
        result = extract_entities("feeling dizzy since yesterday")
        assert result.duration == "1_day"


class TestSeverity:
    """Test severity extraction."""

    def test_severe(self) -> None:
        """Test severe detection."""
        result = extract_entities("severe chest pain")
        assert result.severity == "severe"

    def test_mild(self) -> None:
        """Test mild detection."""
        result = extract_entities("mild headache")
        assert result.severity == "mild"

    def test_pain_scale(self) -> None:
        """Test numeric pain scale."""
        result = extract_entities("pain is 10 out of 10")
        assert result.severity == "severe"


class TestBodyParts:
    """Test body part extraction."""

    def test_chest(self) -> None:
        """Test chest extraction."""
        result = extract_entities("pain in my chest")
        assert "chest" in result.body_parts

    def test_left_arm(self) -> None:
        """Test left arm extraction."""
        result = extract_entities("numbness in left arm")
        assert "left arm" in result.body_parts


class TestComplexCases:
    """Test complex real-world scenarios."""

    def test_cardiac_presentation(self) -> None:
        """Test typical cardiac emergency presentation."""
        result = extract_entities(
            "55 year old male with severe crushing chest pain for 2 hours, "
            "shortness of breath, sweating, pain radiating to left arm"
        )
        assert "chest_pain" in result.symptoms
        assert "shortness_of_breath" in result.symptoms
        assert "sweating" in result.symptoms
        assert result.severity == "severe"
        assert result.duration == "2_hours"

    def test_gi_presentation(self) -> None:
        """Test GI presentation."""
        result = extract_entities(
            "Patient has abdominal pain and nausea for 3 days, no fever"
        )
        assert "abdominal_pain" in result.symptoms
        assert "nausea" in result.symptoms
        assert "fever" in result.negated_symptoms
        assert result.duration == "3_days"
