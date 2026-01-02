"""
Medical Triage API Application.

FastAPI application with triage endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.triage import router as triage_router

app = FastAPI(
    title="Medical Triage API",
    description="AI-powered medical triage with specialty routing and differential diagnosis",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(triage_router)


@app.get("/")
async def root():
    return {
        "name": "Medical Triage API",
        "version": "0.5.0",
        "docs": "/docs",
    }
