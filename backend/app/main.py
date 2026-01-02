"""FastAPI application entry point."""
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.routes import health, triage
from app.api.routes import triage_v2

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Medical Triage AI",
    description="AI-powered medical triage routing system",
    version="0.2.0",
    debug=settings.debug,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(triage.router, prefix="/api", tags=["Triage V1"])
app.include_router(triage_v2.router, prefix="/api", tags=["Triage V2"])


@app.on_event("startup")
async def startup_event() -> None:
    """Log application startup."""
    logger.info(
        "application_startup",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        debug=settings.debug,
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Log application shutdown and cleanup."""
    from app.core.triage_pipeline_v2 import get_triage_pipeline
    pipeline = get_triage_pipeline()
    pipeline.unload()
    logger.info("application_shutdown")
