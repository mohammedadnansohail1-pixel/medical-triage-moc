"""Tests for LLM provider interface."""

import pytest
from app.core.llm_provider import get_llm_provider, OllamaProvider, ClaudeProvider
from app.config import settings


def test_get_llm_provider_ollama() -> None:
    """Test that ollama provider is returned by default."""
    settings.llm_provider = "ollama"
    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)


def test_get_llm_provider_claude() -> None:
    """Test that claude provider is returned when configured."""
    settings.llm_provider = "claude"
    provider = get_llm_provider()
    assert isinstance(provider, ClaudeProvider)
    # Reset
    settings.llm_provider = "ollama"


def test_get_llm_provider_invalid() -> None:
    """Test that invalid provider raises error."""
    settings.llm_provider = "invalid"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_provider()
    # Reset
    settings.llm_provider = "ollama"


@pytest.mark.asyncio
async def test_ollama_health_check_offline() -> None:
    """Test ollama health check when server is offline."""
    provider = OllamaProvider()
    provider.host = "http://localhost:59999"  # Valid but unused port
    result = await provider.health_check()
    assert result is False
