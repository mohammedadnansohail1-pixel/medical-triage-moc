"""Tests for multiagent conversation system."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from app.agents.state import (
    ConversationState,
    create_initial_state,
    escalate_risk,
    RISK_LEVELS,
    MAX_CONVERSATION_TURNS,
)
from app.agents.emergency_agent import quick_emergency_scan, check_emergency
from app.agents.dermatology_agent import is_skin_related
from app.agents.cardiology_agent import is_cardiac_related, check_cardiac_red_flags


class TestConversationState:
    """Test conversation state management."""
    
    def test_create_initial_state(self):
        state = create_initial_state(
            message="I have a headache",
            patient_info={"age": 30, "sex": "male"},
        )
        
        assert state["messages"] == [{"role": "user", "content": "I have a headache"}]
        assert state["symptoms_collected"] == []
        assert state["current_agent"] == "supervisor"
        assert state["risk_level"] == "unknown"
        assert state["triage_complete"] is False
        assert state["turn_count"] == 1
    
    def test_create_initial_state_with_image(self):
        state = create_initial_state(
            message="I have a rash",
            image_base64="base64data",
        )
        
        assert state["image_base64"] == "base64data"
    
    def test_escalate_risk_higher(self):
        assert escalate_risk("unknown", "routine") == "routine"
        assert escalate_risk("routine", "elevated") == "elevated"
        assert escalate_risk("elevated", "urgent") == "urgent"
        assert escalate_risk("urgent", "emergency") == "emergency"
    
    def test_escalate_risk_lower_no_change(self):
        assert escalate_risk("urgent", "routine") == "urgent"
        assert escalate_risk("emergency", "elevated") == "emergency"
    
    def test_risk_level_ordering(self):
        assert RISK_LEVELS["unknown"] < RISK_LEVELS["routine"]
        assert RISK_LEVELS["routine"] < RISK_LEVELS["elevated"]
        assert RISK_LEVELS["elevated"] < RISK_LEVELS["urgent"]
        assert RISK_LEVELS["urgent"] < RISK_LEVELS["emergency"]


class TestEmergencyAgent:
    """Test emergency detection."""
    
    def test_quick_scan_detects_breathing(self):
        assert quick_emergency_scan("I can't breathe") is True
        assert quick_emergency_scan("difficulty breathing") is True
    
    def test_quick_scan_detects_chest_pain(self):
        assert quick_emergency_scan("chest pain radiating to arm") is True
        assert quick_emergency_scan("crushing chest pain") is True
    
    def test_quick_scan_detects_stroke(self):
        assert quick_emergency_scan("face drooping") is True
        assert quick_emergency_scan("slurred speech") is True
        assert quick_emergency_scan("numbness on one side") is True
    
    def test_quick_scan_detects_mental_health(self):
        assert quick_emergency_scan("I want to die") is True
        assert quick_emergency_scan("suicidal thoughts") is True
    
    def test_quick_scan_normal_symptoms(self):
        assert quick_emergency_scan("I have a headache") is False
        assert quick_emergency_scan("my knee hurts") is False
        assert quick_emergency_scan("I have a rash") is False


class TestDermatologyAgent:
    """Test dermatology routing."""
    
    def test_detects_skin_keywords(self):
        assert is_skin_related("I have a rash") is True
        assert is_skin_related("there's a mole on my back") is True
        assert is_skin_related("itchy skin") is True
        assert is_skin_related("red bump on my arm") is True
    
    def test_non_skin_symptoms(self):
        assert is_skin_related("my stomach hurts") is False
        assert is_skin_related("I have a headache") is False
        assert is_skin_related("chest pain") is False


class TestCardiologyAgent:
    """Test cardiology routing."""
    
    def test_detects_cardiac_keywords(self):
        assert is_cardiac_related("chest pain") is True
        assert is_cardiac_related("heart palpitations") is True
        assert is_cardiac_related("shortness of breath") is True
        assert is_cardiac_related("racing heart") is True
    
    def test_non_cardiac_symptoms(self):
        assert is_cardiac_related("I have a rash") is False
        assert is_cardiac_related("knee pain") is False
    
    def test_red_flags_detection(self):
        symptoms = ["chest pain"]
        messages = [{"role": "user", "content": "pain radiates to my left arm"}]
        
        flags = check_cardiac_red_flags(symptoms, messages)
        assert len(flags) > 0
        assert any("arm" in f for f in flags)
    
    def test_no_red_flags(self):
        symptoms = ["mild fatigue"]
        messages = [{"role": "user", "content": "I feel tired"}]
        
        flags = check_cardiac_red_flags(symptoms, messages)
        assert len(flags) == 0


class TestConversationFlow:
    """Test full conversation flows."""
    
    @pytest.mark.asyncio
    async def test_simple_conversation(self):
        from app.agents.supervisor import run_conversation_turn
        
        result = await run_conversation_turn(
            session_id="test-simple",
            message="I have a headache",
            patient_info={"age": 30, "sex": "male"},
        )
        
        assert result["session_id"] == "test-simple"
        assert result["response"]  # Has some response
        assert result["turn_count"] >= 1
    
    @pytest.mark.asyncio
    async def test_emergency_stops_conversation(self):
        from app.agents.supervisor import run_conversation_turn
        
        result = await run_conversation_turn(
            session_id="test-emergency",
            message="I have severe chest pain and can't breathe",
        )
        
        assert result["risk_level"] == "emergency"
        assert result["triage_complete"] is True
        assert "911" in result["response"] or "emergency" in result["response"].lower()
    
    @pytest.mark.asyncio
    async def test_skin_routes_to_dermatology(self):
        from app.agents.supervisor import run_conversation_turn
        
        result = await run_conversation_turn(
            session_id="test-skin",
            message="I have an itchy rash on my arm",
        )
        
        assert result["specialty_hint"] == "dermatology" or result["current_agent"] == "dermatology"
    
    @pytest.mark.asyncio
    async def test_cardiac_routes_to_cardiology(self):
        from app.agents.supervisor import run_conversation_turn
        
        result = await run_conversation_turn(
            session_id="test-cardiac",
            message="I have chest pain and palpitations",
        )
        
        assert result["specialty_hint"] == "cardiology" or result["current_agent"] == "cardiology"


class TestAPIEndpoint:
    """Test API endpoint integration."""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_conversation_endpoint(self, client):
        response = client.post(
            "/api/v1/conversation",
            json={"message": "I have a headache"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "response" in data
        assert "current_agent" in data
    
    def test_conversation_with_patient_info(self, client):
        response = client.post(
            "/api/v1/conversation",
            json={
                "message": "my knee hurts",
                "patient_info": {"age": 45, "sex": "female"},
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["turn_count"] >= 1
    
    def test_conversation_health_endpoint(self, client):
        response = client.get("/api/v1/conversation/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "supervisor" in data["agents"]
        assert "emergency" in data["agents"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
