"""Health check endpoints."""

from typing import Dict

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy"}
