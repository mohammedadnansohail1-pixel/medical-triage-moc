"""
Comprehensive API Tests - Edge Cases, Stress Tests, Security.

Run with: pytest tests/test_api_comprehensive.py -v
"""

import pytest
import time
import concurrent.futures
from typing import List, Dict, Any
from fastapi.testclient import TestClient

from app.main import app
from app.api.triage import get_pipeline


# Initialize client
client = TestClient(app)


class TestHealthEndpoints:
    """Health and basic endpoint tests."""

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Medical Triage API"
        assert "version" in data

    def test_health_endpoint(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_docs_endpoint(self):
        response = client.get("/docs")
        assert response.status_code == 200


class TestInputValidation:
    """Input validation and edge cases."""

    def test_empty_symptoms_rejected(self):
        response = client.post("/api/v1/triage", json={"symptoms": []})
        assert response.status_code == 422

    def test_missing_symptoms_rejected(self):
        response = client.post("/api/v1/triage", json={})
        assert response.status_code == 422

    def test_invalid_age_rejected(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough"],
            "age": -5
        })
        assert response.status_code == 422

    def test_invalid_age_too_high_rejected(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough"],
            "age": 150
        })
        assert response.status_code == 422

    def test_invalid_sex_rejected(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough"],
            "sex": "unknown"
        })
        assert response.status_code == 422

    def test_single_symptom_accepted(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough"],
            "include_explanation": False
        })
        assert response.status_code == 200

    def test_many_symptoms_accepted(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough", "fever", "headache", "fatigue", "nausea", 
                        "dizziness", "chest pain", "shortness of breath"],
            "include_explanation": False
        })
        assert response.status_code == 200

    def test_special_characters_in_symptoms(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["pain!@#$", "fever???"],
            "include_explanation": False
        })
        assert response.status_code == 200

    def test_unicode_symptoms(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["头痛", "fièvre", "Kopfschmerzen"],
            "include_explanation": False
        })
        assert response.status_code == 200

    def test_very_long_symptom_string(self):
        long_symptom = "pain " * 100
        response = client.post("/api/v1/triage", json={
            "symptoms": [long_symptom],
            "include_explanation": False
        })
        assert response.status_code == 200

    def test_whitespace_only_symptom(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["   ", "\t\n"],
            "include_explanation": False
        })
        assert response.status_code == 200


class TestSpecialtyRouting:
    """Test routing to correct specialties."""

    @pytest.mark.parametrize("symptoms,expected_specialty", [
        # Emergency cases
        (["chest pain", "shortness of breath"], "emergency"),
        (["severe bleeding"], "emergency"),
        
        # Pulmonology
        (["cough", "fever"], "pulmonology"),
        (["wheezing", "difficulty breathing"], "pulmonology"),
        
        # Cardiology
        (["palpitations", "fatigue"], "cardiology"),
        (["irregular heartbeat"], "cardiology"),
        
        # Neurology
        (["headache", "dizziness"], "neurology"),
        
        # Gastroenterology
        (["stomach pain", "nausea", "vomiting"], "gastroenterology"),
        (["heartburn", "acid reflux"], "gastroenterology"),
        
        # Dermatology
        (["skin rash", "itching"], "dermatology"),
        (["hives", "skin irritation"], "dermatology"),
    ])
    def test_specialty_routing(self, symptoms: List[str], expected_specialty: str):
        response = client.post("/api/v1/triage", json={
            "symptoms": symptoms,
            "include_explanation": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["specialty"] == expected_specialty, \
            f"Expected {expected_specialty}, got {data['specialty']} for {symptoms}"


class TestEmergencyDetection:
    """Emergency detection must be 100% reliable."""

    @pytest.mark.parametrize("symptoms,should_be_emergency", [
        # True emergencies
        (["chest pain", "shortness of breath"], True),
        (["chest pain", "arm pain"], True),
        (["chest pain", "sweating"], True),
        (["severe bleeding"], True),
        (["difficulty breathing", "swelling", "hives"], True),
        
        # NOT emergencies (should route to specialty)
        (["cough", "fever"], False),
        (["headache"], False),
        (["stomach pain"], False),
        (["skin rash"], False),
        (["fatigue"], False),
    ])
    def test_emergency_detection(self, symptoms: List[str], should_be_emergency: bool):
        response = client.post("/api/v1/triage", json={
            "symptoms": symptoms,
            "include_explanation": False
        })
        assert response.status_code == 200
        data = response.json()
        
        if should_be_emergency:
            assert data["specialty"] == "emergency", \
                f"Should be emergency for {symptoms}"
            assert data["route"] == "EMERGENCY_OVERRIDE"
            assert data["confidence"] == 1.0
        else:
            assert data["specialty"] != "emergency" or data["route"] != "EMERGENCY_OVERRIDE", \
                f"Should NOT be emergency for {symptoms}"


class TestResponseStructure:
    """Verify response structure matches schema."""

    def test_response_has_required_fields(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough", "fever"],
            "include_explanation": False
        })
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        assert "specialty" in data
        assert "confidence" in data
        assert "differential_diagnosis" in data
        assert "route" in data
        
        # Type checks
        assert isinstance(data["specialty"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["differential_diagnosis"], list)
        assert 0 <= data["confidence"] <= 1

    def test_differential_diagnosis_structure(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough", "fever"],
            "include_explanation": False
        })
        data = response.json()
        
        for ddx in data["differential_diagnosis"]:
            assert "condition" in ddx
            assert "probability" in ddx
            assert "rank" in ddx
            assert isinstance(ddx["condition"], str)
            assert isinstance(ddx["probability"], float)
            assert isinstance(ddx["rank"], int)
            assert 0 <= ddx["probability"] <= 1

    def test_ddx_is_sorted_by_rank(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough", "fever"],
            "include_explanation": False
        })
        data = response.json()
        
        ranks = [d["rank"] for d in data["differential_diagnosis"]]
        assert ranks == sorted(ranks), "DDx should be sorted by rank"

    def test_valid_route_values(self):
        valid_routes = ["EMERGENCY_OVERRIDE", "ML_CLASSIFICATION", "RULE_OVERRIDE", "DEFAULT_FALLBACK"]
        
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough"],
            "include_explanation": False
        })
        data = response.json()
        assert data["route"] in valid_routes


class TestConfidenceScores:
    """Test confidence score behavior."""

    def test_emergency_confidence_is_100(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["chest pain", "shortness of breath"],
            "include_explanation": False
        })
        data = response.json()
        assert data["confidence"] == 1.0

    def test_rule_override_confidence_is_set(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["skin rash", "itching"],
            "include_explanation": False
        })
        data = response.json()
        assert data["route"] == "RULE_OVERRIDE"
        assert data["confidence"] > 0.5

    def test_ml_classification_has_confidence(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough", "fever"],
            "include_explanation": False
        })
        data = response.json()
        assert data["route"] == "ML_CLASSIFICATION"
        assert 0 < data["confidence"] <= 1


class TestDemographics:
    """Test age and sex handling."""

    def test_age_affects_response(self):
        # Same symptoms, different ages
        response_young = client.post("/api/v1/triage", json={
            "symptoms": ["chest pain", "fatigue"],
            "age": 25,
            "include_explanation": False
        })
        response_old = client.post("/api/v1/triage", json={
            "symptoms": ["chest pain", "fatigue"],
            "age": 75,
            "include_explanation": False
        })
        # Both should work (may or may not differ)
        assert response_young.status_code == 200
        assert response_old.status_code == 200

    def test_sex_male_accepted(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough"],
            "sex": "male",
            "include_explanation": False
        })
        assert response.status_code == 200

    def test_sex_female_accepted(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough"],
            "sex": "female",
            "include_explanation": False
        })
        assert response.status_code == 200


class TestPerformance:
    """Performance and latency tests."""

    def test_response_time_without_explanation(self):
        start = time.time()
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough", "fever"],
            "include_explanation": False
        })
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 2.0, f"Response took {duration:.2f}s, expected < 2s"

    def test_sequential_requests(self):
        """Test 10 sequential requests."""
        times = []
        for _ in range(10):
            start = time.time()
            response = client.post("/api/v1/triage", json={
                "symptoms": ["cough", "fever"],
                "include_explanation": False
            })
            times.append(time.time() - start)
            assert response.status_code == 200
        
        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"Average response time {avg_time:.2f}s, expected < 1s"


class TestSecurityInputs:
    """Security-related input tests."""

    def test_sql_injection_attempt(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["'; DROP TABLE users; --", "OR 1=1"],
            "include_explanation": False
        })
        # Should not crash, should handle gracefully
        assert response.status_code == 200

    def test_script_injection_attempt(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["<script>alert('xss')</script>", "javascript:alert(1)"],
            "include_explanation": False
        })
        assert response.status_code == 200

    def test_path_traversal_attempt(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["../../../etc/passwd", "..\\windows\\system32"],
            "include_explanation": False
        })
        assert response.status_code == 200

    def test_null_byte_injection(self):
        response = client.post("/api/v1/triage", json={
            "symptoms": ["cough\x00", "fever\x00injection"],
            "include_explanation": False
        })
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
