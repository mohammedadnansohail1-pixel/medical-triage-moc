"""
Medical Triage API Application.
FastAPI application with triage endpoints.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.triage import router as triage_router
from app.api.image_triage import router as image_triage_router
from app.api.conversation import router as conversation_router

app = FastAPI(
    title="Medical Triage API",
    description="AI-powered medical triage with specialty routing, differential diagnosis, and skin image analysis",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(triage_router)
app.include_router(image_triage_router)
app.include_router(conversation_router)


@app.get("/")
async def root():
    return {
        "name": "Medical Triage API",
        "version": "0.6.0",
        "docs": "/docs",
        "endpoints": {
            "triage": "/api/v1/triage",
            "image_triage": "/api/v1/triage/image",
            "health": "/api/v1/health",
            "conversation": "/api/v1/conversation",
        },
    }
