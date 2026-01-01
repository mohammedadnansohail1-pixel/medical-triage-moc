"""
Abstracted LLM interface - switch providers via LLM_PROVIDER env var.
Supports: ollama (local), claude (Anthropic), openai
"""

from abc import ABC, abstractmethod
from typing import Optional

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama (uses GPU)."""

    def __init__(self) -> None:
        self.host = settings.ollama_host
        self.model = settings.llm_model

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using local Ollama."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                result = response.json()["response"]
                logger.info(
                    "ollama_generate_success",
                    model=self.model,
                    prompt_length=len(prompt),
                    response_length=len(result),
                )
                return result
            except httpx.HTTPError as e:
                logger.error("ollama_generate_error", error=str(e))
                raise

    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API (cloud fallback)."""

    def __init__(self) -> None:
        self.api_key = settings.anthropic_api_key
        self.model = settings.llm_model or "claude-sonnet-4-20250514"

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response using Claude API."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                result = response.json()["content"][0]["text"]
                logger.info(
                    "claude_generate_success",
                    model=self.model,
                    prompt_length=len(prompt),
                    response_length=len(result),
                )
                return result
            except httpx.HTTPError as e:
                logger.error("claude_generate_error", error=str(e))
                raise

    async def health_check(self) -> bool:
        """Check if Claude API is accessible."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                return response.status_code == 200
        except httpx.HTTPError:
            return False


def get_llm_provider() -> LLMProvider:
    """Factory function to get LLM provider based on config."""
    provider_map = {
        "ollama": OllamaProvider,
        "claude": ClaudeProvider,
    }

    provider_class = provider_map.get(settings.llm_provider)
    if not provider_class:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    logger.info("llm_provider_initialized", provider=settings.llm_provider)
    return provider_class()
