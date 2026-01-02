"""
Supervisor Agent - Orchestrates the multiagent conversation.

Responsibilities:
- Route user messages to appropriate specialist agents
- Ask clarifying questions when needed
- Synthesize responses
- Manage conversation flow
"""

from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .state import (
    ConversationState,
    create_initial_state,
    escalate_risk,
    MAX_CONVERSATION_TURNS,
)
from .emergency_agent import run_emergency_check, quick_emergency_scan
from .triage_agent import run_triage_node
from .dermatology_agent import run_dermatology_node, is_skin_related
from .cardiology_agent import run_cardiology_node, is_cardiac_related


# Initialize LLM
def get_supervisor_llm():
    """Get the supervisor LLM instance."""
    return ChatOllama(
        model="llama3.1:8b",
        temperature=0.3,  # Slightly creative for natural conversation
    )


# Supervisor system prompt
SUPERVISOR_SYSTEM_PROMPT = """You are a medical triage assistant helping patients understand their symptoms.

Your role is to:
1. Gather information about the patient's symptoms through natural conversation
2. Ask clarifying questions to understand the full picture
3. Route to specialists when appropriate
4. Never diagnose - only help patients understand when/where to seek care

IMPORTANT RULES:
- Always be empathetic and professional
- If symptoms sound serious (chest pain, difficulty breathing, severe headache), express appropriate concern
- Extract specific symptoms from the conversation
- Ask ONE question at a time
- Keep responses concise (2-3 sentences max for questions)

Based on the conversation, decide your next action:
- "ask_clarification": Need more info about symptoms
- "route_dermatology": Skin-related symptoms detected
- "route_cardiology": Heart-related symptoms detected  
- "run_triage": Have enough info for full assessment
- "end_conversation": Triage complete or max turns reached

Respond in JSON format:
{
    "action": "<action_name>",
    "extracted_symptoms": ["symptom1", "symptom2"],
    "response": "Your message to the patient",
    "reasoning": "Brief explanation of your decision"
}
"""


def parse_supervisor_response(response_text: str) -> dict:
    """Parse supervisor LLM response, handling various formats."""
    import json
    import re
    
    # Try direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text
    json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Fallback: extract what we can
    action = "ask_clarification"
    if "dermatology" in response_text.lower():
        action = "route_dermatology"
    elif "cardiology" in response_text.lower() or "heart" in response_text.lower():
        action = "route_cardiology"
    elif "triage" in response_text.lower() or "assessment" in response_text.lower():
        action = "run_triage"
    
    return {
        "action": action,
        "extracted_symptoms": [],
        "response": response_text[:500],  # Truncate if needed
        "reasoning": "Fallback parsing",
    }


def supervisor_node(state: ConversationState) -> dict:
    """
    Main supervisor node - decides routing and generates responses.
    """
    messages = state.get("messages", [])
    symptoms = state.get("symptoms_collected", [])
    turn_count = state.get("turn_count", 0)
    image_base64 = state.get("image_base64")
    
    # Check turn limit
    if turn_count >= MAX_CONVERSATION_TURNS:
        return {
            "messages": [{
                "role": "assistant",
                "content": (
                    "We've been chatting for a while. Based on what you've shared, "
                    "I recommend scheduling an appointment with a healthcare provider "
                    "for a proper evaluation. Would you like me to provide a summary "
                    "of the symptoms we discussed?"
                ),
            }],
            "triage_complete": True,
            "current_agent": "supervisor",
        }
    
    # Quick checks for routing
    latest_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user_msg = m.get("content", "")
            break
    
    # Emergency scan (fast, rule-based)
    if quick_emergency_scan(latest_user_msg):
        return {
            "current_agent": "emergency",
        }
    
    # Image provided + skin context -> dermatology
    if image_base64 and (is_skin_related(latest_user_msg) or is_skin_related(" ".join(symptoms))):
        return {
            "current_agent": "dermatology",
        }
    
    # Build conversation for LLM
    llm = get_supervisor_llm()
    
    # Format messages for LLM
    llm_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]
    
    # Add conversation history (last 10 messages)
    for m in messages[-10:]:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            llm_messages.append(HumanMessage(content=content))
        else:
            llm_messages.append(AIMessage(content=content))
    
    # Add context about current state
    context = f"""
Current state:
- Symptoms collected: {symptoms}
- Turn count: {turn_count}
- Has image: {bool(image_base64)}
- Patient info: {state.get('patient_info', {})}

Decide your next action and respond to the patient.
"""
    llm_messages.append(HumanMessage(content=context))
    
    # Get LLM response
    try:
        response = llm.invoke(llm_messages)
        parsed = parse_supervisor_response(response.content)
    except Exception as e:
        # Fallback on error
        parsed = {
            "action": "ask_clarification",
            "extracted_symptoms": [],
            "response": "I'd like to understand your symptoms better. Can you tell me more about what you're experiencing?",
            "reasoning": f"Error fallback: {str(e)}",
        }
    
    # Extract new symptoms
    new_symptoms = parsed.get("extracted_symptoms", [])
    if new_symptoms:
        symptoms = list(set(symptoms + new_symptoms))
    
    # Determine next agent based on action
    action = parsed.get("action", "ask_clarification")
    
    result = {
        "symptoms_collected": symptoms,
        "turn_count": turn_count + 1,
    }
    
    if action == "route_dermatology" or is_skin_related(latest_user_msg):
        result["current_agent"] = "dermatology"
    elif action == "route_cardiology" or is_cardiac_related(latest_user_msg):
        result["current_agent"] = "cardiology"
    elif action == "run_triage" and len(symptoms) >= 2:
        result["current_agent"] = "triage"
    elif action == "end_conversation":
        result["triage_complete"] = True
        result["current_agent"] = "supervisor"
        result["messages"] = [{
            "role": "assistant",
            "content": parsed.get("response", "Thank you for chatting. Please consult a healthcare provider."),
        }]
    else:
        # Continue conversation
        result["current_agent"] = "supervisor"
        result["messages"] = [{
            "role": "assistant", 
            "content": parsed.get("response", "Can you tell me more about your symptoms?"),
        }]
    
    return result


def route_to_agent(state: ConversationState) -> str:
    """Determine which agent node to route to."""
    current = state.get("current_agent", "supervisor")
    triage_complete = state.get("triage_complete", False)
    
    if triage_complete:
        return "end"
    
    return current


def build_conversation_graph() -> StateGraph:
    """Build the LangGraph conversation graph."""
    
    # Create graph with state schema
    graph = StateGraph(ConversationState)
    
    # Add nodes
    graph.add_node("emergency", run_emergency_check)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("triage", run_triage_node)
    graph.add_node("dermatology", run_dermatology_node)
    graph.add_node("cardiology", run_cardiology_node)
    
    # Entry point - always check emergency first
    graph.add_edge(START, "emergency")
    
    # After emergency check, route based on result
    graph.add_conditional_edges(
        "emergency",
        lambda s: "end" if s.get("triage_complete") else "supervisor",
        {
            "end": END,
            "supervisor": "supervisor",
        }
    )
    
    # Supervisor routes to specialists or ends
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "supervisor": END,  # Response ready, wait for user
            "dermatology": "dermatology",
            "cardiology": "cardiology",
            "triage": "triage",
            "emergency": "emergency",
            "end": END,
        }
    )
    
    # Specialist agents return to end (response ready)
    graph.add_edge("dermatology", END)
    graph.add_edge("cardiology", END)
    graph.add_edge("triage", END)
    
    return graph


# Compiled graph with memory
_conversation_graph = None
_memory_saver = None


def get_conversation_graph():
    """Get compiled conversation graph with memory."""
    global _conversation_graph, _memory_saver
    
    if _conversation_graph is None:
        graph = build_conversation_graph()
        _memory_saver = MemorySaver()
        _conversation_graph = graph.compile(checkpointer=_memory_saver)
    
    return _conversation_graph


async def run_conversation_turn(
    session_id: str,
    message: str,
    patient_info: dict | None = None,
    image_base64: str | None = None,
) -> dict:
    """
    Run a single conversation turn.
    
    Args:
        session_id: Unique session identifier for conversation continuity
        message: User's message
        patient_info: Optional patient demographics
        image_base64: Optional image data
        
    Returns:
        Response dict with assistant message and state info
    """
    graph = get_conversation_graph()
    
    # Configuration for this thread
    config = {"configurable": {"thread_id": session_id}}
    
    # Get current state or create initial
    try:
        current_state = graph.get_state(config)
        if current_state.values:
            # Continue conversation
            input_state = {
                "messages": [{"role": "user", "content": message}],
                "image_base64": image_base64,
            }
            if patient_info:
                input_state["patient_info"] = patient_info
        else:
            # New conversation
            input_state = create_initial_state(
                message=message,
                patient_info=patient_info,
                image_base64=image_base64,
            )
    except Exception:
        # New conversation
        input_state = create_initial_state(
            message=message,
            patient_info=patient_info,
            image_base64=image_base64,
        )
    
    # Run graph
    result = await graph.ainvoke(input_state, config)
    
    # Extract response
    messages = result.get("messages", [])
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    latest_response = assistant_messages[-1]["content"] if assistant_messages else "I'm here to help. What symptoms are you experiencing?"
    
    return {
        "session_id": session_id,
        "response": latest_response,
        "current_agent": result.get("current_agent", "supervisor"),
        "symptoms_collected": result.get("symptoms_collected", []),
        "risk_level": result.get("risk_level", "unknown"),
        "triage_complete": result.get("triage_complete", False),
        "turn_count": result.get("turn_count", 1),
        "specialty_hint": result.get("specialty_hint"),
        "triage_result": result.get("triage_result"),
        "warnings": result.get("warnings", []),
    }


# Synchronous wrapper for non-async contexts
def run_conversation_turn_sync(
    session_id: str,
    message: str,
    patient_info: dict | None = None,
    image_base64: str | None = None,
) -> dict:
    """Synchronous wrapper for run_conversation_turn."""
    import asyncio
    return asyncio.run(run_conversation_turn(
        session_id=session_id,
        message=message,
        patient_info=patient_info,
        image_base64=image_base64,
    ))
