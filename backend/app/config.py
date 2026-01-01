"""Application configuration with environment variable loading."""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

# Find .env in project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields in .env
    )

    # LLM Configuration
    llm_provider: Literal["ollama", "claude", "openai"] = "ollama"
    llm_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Database
    postgres_password: str = "triage123"
    database_url: str = "postgresql://postgres:triage123@localhost:5432/triage"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "triage123"
    redis_url: str = "redis://localhost:6379/0"

    # App
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


settings = Settings()
