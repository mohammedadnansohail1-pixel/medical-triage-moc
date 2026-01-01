"""Application configuration with environment variable loading."""

from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    llm_provider: Literal["ollama", "claude", "openai"] = "ollama"
    llm_model: str = "mistral:7b-instruct-q8_0"
    ollama_host: str = "http://localhost:11434"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Database
    database_url: str = "postgresql://postgres:triage123@localhost:5432/triage"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "triage123"
    redis_url: str = "redis://localhost:6379/0"

    # App
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
