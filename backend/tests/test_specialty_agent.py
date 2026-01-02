"""
Tests for Specialty Agents.

Run: pytest tests/test_specialty_agent.py -v
"""

import pytest
from pathlib import Path

from app.core.specialty_agent import (
    SpecialtyAgent,
    SpecialtyAgentManager,
    get_specialty_manager,
)


@pytest.fixture
def model_path():
    return Path("/home/adnan21/projects/medical-triage-moc/data/ddxplus/condition_model.json")


@pytest.fixture
def manager(model_path):
    return SpecialtyAgentManager(model_path)


class TestSpecialtyAgentManager:
    """Test agent manager."""

    def test_load_model(self, manager):
        """Should load model data."""
        assert manager.available_specialties
        assert len(manager.available_specialties) >= 6

    def test_available_specialties(self, manager):
        """Should have expected specialties."""
        specs = manager.available_specialties
        assert "pulmonology" in specs
        assert "cardiology" in specs
        assert "general_medicine" in specs

    def test_get_agent(self, manager):
        """Should create agent for specialty."""
        agent = manager.get_agent("pulmonology")
        assert isinstance(agent, SpecialtyAgent)
        assert agent.specialty == "pulmonology"

    def test_get_agent_cached(self, manager):
        """Should return same agent instance."""
        agent1 = manager.get_agent("cardiology")
        agent2 = manager.get_agent("cardiology")
        assert agent1 is agent2

    def test_invalid_specialty(self, manager):
        """Should raise for invalid specialty."""
        with pytest.raises(ValueError):
            manager.get_agent("invalid_specialty")


class TestSpecialtyAgent:
    """Test individual agent."""

    def test_diagnose_returns_list(self, manager):
        """Should return list of tuples."""
        result = manager.diagnose(
            specialty="pulmonology",
            symptom_codes=["E_201", "E_91"],
            top_k=5
        )
        assert isinstance(result, list)
        assert len(result) <= 5
        assert all(isinstance(item, tuple) for item in result)

    def test_diagnose_sorted_by_probability(self, manager):
        """Results should be sorted by probability desc."""
        result = manager.diagnose(
            specialty="pulmonology",
            symptom_codes=["E_201", "E_91", "E_50"],
            top_k=5
        )
        probs = [p for _, p in result]
        assert probs == sorted(probs, reverse=True)

    def test_diagnose_probabilities_sum_approximately_one(self, manager):
        """Top-k probabilities should sum close to 1 if k covers all."""
        agent = manager.get_agent("emergency")  # Only 4 conditions
        result = agent.diagnose(["E_91"], top_k=10)
        total = sum(p for _, p in result)
        assert 0.99 <= total <= 1.01

    def test_diagnose_different_symptoms_different_results(self, manager):
        """Different symptoms should give different rankings."""
        result1 = manager.diagnose("cardiology", ["E_14"])  # Chest pain
        result2 = manager.diagnose("cardiology", ["E_49"])  # Palpitations
        
        top1_cond1 = result1[0][0] if result1 else None
        top1_cond2 = result2[0][0] if result2 else None
        
        # May or may not be different, but probabilities should differ
        probs1 = {c: p for c, p in result1}
        probs2 = {c: p for c, p in result2}
        assert probs1 != probs2


class TestPulmonologyAgent:
    """Test pulmonology-specific cases."""

    def test_respiratory_symptoms(self, manager):
        """Respiratory symptoms should give respiratory conditions."""
        result = manager.diagnose(
            specialty="pulmonology",
            symptom_codes=["E_201", "E_77", "E_91"],  # cough, productive cough, fever
            top_k=3
        )
        conditions = [c for c, _ in result]
        respiratory = ["Pneumonia", "Bronchitis", "URTI", "Influenza", "Tuberculosis"]
        assert any(c in respiratory for c in conditions)


class TestCardiologyAgent:
    """Test cardiology-specific cases."""

    def test_chest_pain_symptoms(self, manager):
        """Chest pain should suggest cardiac conditions."""
        result = manager.diagnose(
            specialty="cardiology",
            symptom_codes=["E_14", "E_49"],  # chest pain at rest, palpitations
            top_k=3
        )
        conditions = [c for c, _ in result]
        cardiac = ["Unstable angina", "Possible NSTEMI / STEMI", "Atrial fibrillation", "Pericarditis"]
        assert any(c in cardiac for c in conditions)


class TestIntegration:
    """Integration tests."""

    def test_all_specialties_work(self, manager):
        """All specialties should return results."""
        for specialty in manager.available_specialties:
            result = manager.diagnose(specialty, ["E_91"], top_k=3)
            assert len(result) > 0, f"No results for {specialty}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
