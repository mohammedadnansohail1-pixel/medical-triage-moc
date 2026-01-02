"""
Conversation API - Multiagent symptom checker endpoint.

Provides a conversational interface for medical triage using
LangGraph-based multiagent orchestration.
"""

from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.supervisor import run_conversation_turn


router = APIRouter(prefix="/api/v1", tags=["conversation"])


class PatientInfoRequest(BaseModel):
    """Patient demographic information."""
    age: int | None = Field(None, ge=0, le=120, description="Patient age")
    sex: Literal["male", "female"] | None = Field(None, description="Patient sex")


class ConversationRequest(BaseModel):
    """Request for a conversation turn."""
    session_id: str | None = Field(
        None, 
        description="Session ID for conversation continuity. Generated if not provided."
    )
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="User's message"
    )
    patient_info: PatientInfoRequest | None = Field(
        None,
        description="Optional patient demographics"
    )
    image_base64: str | None = Field(
        None,
        description="Optional base64-encoded image for skin analysis"
    )


class ConversationResponse(BaseModel):
    """Response from a conversation turn."""
    session_id: str = Field(..., description="Session ID for continuity")
    response: str = Field(..., description="Assistant's response")
    current_agent: str = Field(..., description="Agent that handled this turn")
    symptoms_collected: list[str] = Field(
        default_factory=list,
        description="Symptoms identified so far"
    )
    risk_level: str = Field(
        default="unknown",
        description="Current risk assessment"
    )
    triage_complete: bool = Field(
        default=False,
        description="Whether triage is complete"
    )
    turn_count: int = Field(default=1, description="Number of conversation turns")
    specialty_hint: str | None = Field(
        None,
        description="Suggested medical specialty"
    )
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings or notes"
    )


@router.post("/conversation", response_model=ConversationResponse)
async def conversation_turn(request: ConversationRequest) -> ConversationResponse:
    """
    Process a single conversation turn in the symptom checker.
    
    This endpoint enables multi-turn conversations where a supervisor agent
    orchestrates specialist agents (emergency, triage, dermatology, cardiology)
    to gather symptoms and provide triage recommendations.
    
    **Example flow:**
    
    1. User: "My chest hurts"
    2. Assistant: "I'd like to understand better. Is the pain sharp or dull?"
    3. User: "It's sharp and goes to my left arm"
    4. Assistant: "⚠️ These symptoms could indicate a cardiac emergency..."
    
    **Session Management:**
    - Omit `session_id` for a new conversation
    - Include `session_id` from previous response to continue
    
    **Image Support:**
    - For skin-related symptoms, include `image_base64` for AI analysis
    - Supported formats: JPEG, PNG, WebP
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Convert patient info
        patient_info = None
        if request.patient_info:
            patient_info = {
                "age": request.patient_info.age,
                "sex": request.patient_info.sex,
            }
        
        # Run conversation turn
        result = await run_conversation_turn(
            session_id=session_id,
            message=request.message,
            patient_info=patient_info,
            image_base64=request.image_base64,
        )
        
        # Generate suggested actions based on risk level
        suggested_actions = []
        risk_level = result.get("risk_level", "unknown")
        triage_complete = result.get("triage_complete", False)
        
        if triage_complete:
            if risk_level == "emergency":
                suggested_actions = [
                    "Call 911 immediately",
                    "Do not drive yourself",
                    "Follow dispatcher instructions",
                ]
            elif risk_level == "urgent":
                suggested_actions = [
                    "Seek medical attention within 24 hours",
                    "Visit urgent care or ER if symptoms worsen",
                    "Prepare list of symptoms for your doctor",
                ]
            elif risk_level == "elevated":
                suggested_actions = [
                    "Schedule appointment within a few days",
                    "Monitor symptoms",
                    "Note any changes",
                ]
            else:
                suggested_actions = [
                    "Schedule routine appointment",
                    "Continue monitoring",
                ]
        
        return ConversationResponse(
            session_id=result["session_id"],
            response=result["response"],
            current_agent=result["current_agent"],
            symptoms_collected=result["symptoms_collected"],
            risk_level=result["risk_level"],
            triage_complete=result["triage_complete"],
            turn_count=result["turn_count"],
            specialty_hint=result.get("specialty_hint"),
            suggested_actions=suggested_actions,
            warnings=result.get("warnings", []),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Conversation error: {str(e)}"
        )


@router.post("/conversation/reset")
async def reset_conversation(session_id: str) -> dict:
    """
    Reset a conversation session.
    
    This clears the conversation history for the given session,
    allowing the user to start fresh.
    """
    # Note: With MemorySaver, we'd need to clear the checkpoint
    # For now, just acknowledge - client should use new session_id
    return {
        "status": "ok",
        "message": f"Session {session_id} marked for reset. Use a new session_id to start fresh.",
    }


@router.get("/conversation/health")
async def conversation_health() -> dict:
    """Health check for conversation system."""
    try:
        # Quick test that graph builds
        from agents.supervisor import get_conversation_graph
        graph = get_conversation_graph()
        
        return {
            "status": "healthy",
            "graph_nodes": list(graph.get_graph().nodes.keys()),
            "agents": ["supervisor", "emergency", "triage", "dermatology", "cardiology"],
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
