"""
Tests for skin lesion classifier with 4-tier risk stratification.
"""
import pytest
import base64
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.core.skin_classifier import (
    SkinLesionClassifier, 
    get_skin_classifier,
    CLASS_INFO,
    TIER_INFO,
    RiskTier,
    THRESHOLDS,
)

client = TestClient(app)


class TestSkinClassifier:
    """Test skin classifier module."""

    def test_classifier_loads(self):
        """Test classifier loads successfully."""
        classifier = SkinLesionClassifier()
        classifier.load()
        assert classifier._loaded is True
        assert classifier.model is not None

    def test_classifier_singleton(self):
        """Test singleton pattern."""
        c1 = get_skin_classifier()
        c2 = get_skin_classifier()
        assert c1 is c2

    def test_predict_returns_expected_fields(self):
        """Test prediction returns all expected fields."""
        classifier = get_skin_classifier()
        image = Image.new('RGB', (224, 224), color=(200, 150, 140))
        
        result = classifier.predict(image)
        
        assert "prediction" in result
        assert "prediction_label" in result
        assert "confidence" in result
        assert "all_predictions" in result
        assert "probability_summary" in result
        assert "risk_assessment" in result
        assert "disclaimer" in result

    def test_risk_assessment_fields(self):
        """Test risk assessment has all required fields."""
        classifier = get_skin_classifier()
        image = Image.new('RGB', (224, 224), color=(200, 150, 140))
        
        result = classifier.predict(image)
        risk = result["risk_assessment"]
        
        assert "tier" in risk
        assert "tier_display" in risk
        assert "color" in risk
        assert "timeframe" in risk
        assert "message" in risk
        assert "action" in risk
        assert "reasons" in risk

    def test_probability_summary_fields(self):
        """Test probability summary has required fields."""
        classifier = get_skin_classifier()
        image = Image.new('RGB', (224, 224), color=(200, 150, 140))
        
        result = classifier.predict(image)
        probs = result["probability_summary"]
        
        assert "benign" in probs
        assert "precancer" in probs
        assert "cancer_total" in probs
        assert "melanoma" in probs

    def test_tier_is_valid(self):
        """Test tier is a valid RiskTier value."""
        classifier = get_skin_classifier()
        image = Image.new('RGB', (224, 224), color=(200, 150, 140))
        
        result = classifier.predict(image)
        
        valid_tiers = [t.value for t in RiskTier]
        assert result["risk_assessment"]["tier"] in valid_tiers

    def test_confidence_range(self):
        """Test confidence is in valid range."""
        classifier = get_skin_classifier()
        image = Image.new('RGB', (224, 224), color=(180, 140, 130))
        
        result = classifier.predict(image)
        
        assert 0.0 <= result["confidence"] <= 1.0

    def test_valid_class_prediction(self):
        """Test prediction is a valid class."""
        classifier = get_skin_classifier()
        image = Image.new('RGB', (224, 224), color=(190, 150, 140))
        
        result = classifier.predict(image)
        
        assert result["prediction"] in CLASS_INFO

    def test_rgb_conversion(self):
        """Test non-RGB images are converted."""
        classifier = get_skin_classifier()
        image = Image.new('L', (224, 224), color=128)
        
        result = classifier.predict(image)
        assert result["prediction"] in CLASS_INFO

    def test_all_tiers_have_info(self):
        """Test all risk tiers have display info."""
        for tier in RiskTier:
            assert tier in TIER_INFO
            info = TIER_INFO[tier]
            assert "display_name" in info
            assert "color" in info
            assert "message" in info

    def test_thresholds_defined(self):
        """Test safety thresholds are defined."""
        assert "benign_high_confidence" in THRESHOLDS
        assert "melanoma_alert" in THRESHOLDS
        assert THRESHOLDS["melanoma_alert"] < THRESHOLDS["cancer_low"]

    def test_classify_with_context_adds_fields(self):
        """Test context classification adds patient info."""
        classifier = get_skin_classifier()
        image = Image.new('RGB', (224, 224), color=(200, 150, 140))
        
        result = classifier.classify_with_context(
            image=image, 
            symptoms=["itching"],
            age=55,
            sex="male"
        )
        
        assert "context" in result
        assert result["context"]["age"] == 55
        assert result["context"]["symptoms"] == ["itching"]


class TestImageTriageAPI:
    """Test image triage API endpoints."""

    @staticmethod
    def create_test_image(color=(200, 150, 140)) -> bytes:
        """Create test image bytes."""
        image = Image.new('RGB', (224, 224), color=color)
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        return buffer.read()

    def test_triage_image_success(self):
        """Test successful image triage."""
        image_bytes = self.create_test_image()
        
        response = client.post(
            "/api/v1/triage/image",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["specialty"] == "dermatology"
        assert "image_analysis" in data
        assert "risk_assessment" in data["image_analysis"]

    def test_triage_with_metadata(self):
        """Test triage with patient metadata."""
        image_bytes = self.create_test_image()
        
        response = client.post(
            "/api/v1/triage/image",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            data={"symptoms": "itching, growing", "age": 60, "sex": "female"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["patient_age"] == 60
        assert "itching" in data["symptoms_provided"]

    def test_triage_response_has_tier(self):
        """Test response includes tier information."""
        image_bytes = self.create_test_image()
        
        response = client.post(
            "/api/v1/triage/image",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )
        
        data = response.json()
        risk = data["image_analysis"]["risk_assessment"]
        
        assert "tier" in risk
        assert "color" in risk
        assert "message" in risk
        assert risk["color"] in ["green", "yellow", "orange", "red"]

    def test_triage_image_png(self):
        """Test PNG image upload."""
        image = Image.new('RGB', (224, 224), color=(180, 140, 130))
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        response = client.post(
            "/api/v1/triage/image",
            files={"image": ("test.png", buffer.read(), "image/png")},
        )
        
        assert response.status_code == 200

    def test_triage_invalid_type(self):
        """Test rejection of invalid image type."""
        response = client.post(
            "/api/v1/triage/image",
            files={"image": ("test.gif", b"fake", "image/gif")},
        )
        assert response.status_code == 400

    def test_triage_image_too_small(self):
        """Test rejection of small image."""
        image = Image.new('RGB', (30, 30))
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        response = client.post(
            "/api/v1/triage/image",
            files={"image": ("small.jpg", buffer.read(), "image/jpeg")},
        )
        assert response.status_code == 400

    def test_base64_endpoint(self):
        """Test base64 image endpoint."""
        image_bytes = self.create_test_image()
        b64 = base64.b64encode(image_bytes).decode()
        
        response = client.post(
            "/api/v1/triage/image/base64",
            json={"image_base64": b64, "age": 45},
        )
        
        assert response.status_code == 200
        assert response.json()["patient_age"] == 45

    def test_base64_with_data_url(self):
        """Test base64 with data URL prefix."""
        image_bytes = self.create_test_image()
        b64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"
        
        response = client.post(
            "/api/v1/triage/image/base64",
            json={"image_base64": b64},
        )
        assert response.status_code == 200

    def test_skin_classes_endpoint(self):
        """Test skin classes info endpoint."""
        response = client.get("/api/v1/skin-classes")
        
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "risk_tiers" in data
        assert "MEL" in data["urgent_cancer_classes"]


class TestResponseStructure:
    """Test complete response structure."""

    def test_full_response_structure(self):
        """Test all fields in response."""
        image = Image.new('RGB', (224, 224), color=(200, 150, 140))
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        response = client.post(
            "/api/v1/triage/image",
            files={"image": ("test.jpg", buffer.read(), "image/jpeg")},
        )
        
        data = response.json()
        analysis = data["image_analysis"]
        
        # Top-level
        assert "specialty" in data
        assert "route" in data
        assert "symptoms_provided" in data
        
        # Analysis
        assert "prediction" in analysis
        assert "confidence" in analysis
        assert "probability_summary" in analysis
        assert "risk_assessment" in analysis
        assert "disclaimer" in analysis
        
        # Probability summary
        probs = analysis["probability_summary"]
        assert "benign" in probs
        assert "cancer_total" in probs
        
        # Risk assessment
        risk = analysis["risk_assessment"]
        assert "tier" in risk
        assert "message" in risk
        assert "reasons" in risk

    def test_disclaimer_present(self):
        """Test disclaimer is always present."""
        image = Image.new('RGB', (224, 224))
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        response = client.post(
            "/api/v1/triage/image",
            files={"image": ("test.jpg", buffer.read(), "image/jpeg")},
        )
        
        disclaimer = response.json()["image_analysis"]["disclaimer"]
        assert "NOT a medical diagnosis" in disclaimer
