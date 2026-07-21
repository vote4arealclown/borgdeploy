"""Tests for the hybrid LLM escalation strategy."""
from __future__ import annotations

import pytest

from borg.llm_hybrid import HybridLLM


@pytest.fixture
def local_forecast_low_confidence(monkeypatch):
    """Mock the local LLM to return a low-confidence forecast."""

    class FakeLocal:
        async def analyze_market(self, symbol: str, summary: str) -> dict:
            return {
                "direction": "up",
                "confidence": 45.0,
                "key_signals": ["weak signal"],
                "analysis": "uncertain",
                "model_used": "fake-local",
            }

        async def embed(self, text: str) -> list[float]:
            return [0.0] * 768

    return FakeLocal()


@pytest.mark.asyncio
async def test_hybrid_no_escalation_when_confident(monkeypatch) -> None:
    """A high-confidence local forecast should not trigger remote escalation."""

    class FakeLocal:
        async def analyze_market(self, symbol: str, summary: str) -> dict:
            return {
                "direction": "down",
                "confidence": 80.0,
                "key_signals": ["strong signal"],
                "analysis": "confident",
                "model_used": "fake-local",
            }

    hybrid = HybridLLM(local_llm=FakeLocal(), confidence_threshold=65.0, openai_api_key="sk-test")
    result = await hybrid.analyze_market("EURUSD", "summary")

    assert result["direction"] == "down"
    assert result["confidence"] == 80.0
    assert result["overridden"] is False
    assert result["model_used"] == "fake-local"


@pytest.mark.asyncio
async def test_hybrid_no_remote_without_key(local_forecast_low_confidence) -> None:
    """Without an API key, low-confidence forecasts fall back to local."""
    hybrid = HybridLLM(
        local_llm=local_forecast_low_confidence,
        confidence_threshold=65.0,
        openai_api_key=None,
    )
    result = await hybrid.analyze_market("EURUSD", "summary")

    assert result["direction"] == "up"
    assert result["confidence"] == 45.0
    assert result["overridden"] is False


@pytest.mark.asyncio
async def test_hybrid_remote_overrides_disagreement(local_forecast_low_confidence, monkeypatch) -> None:
    """When the remote model disagrees, the forecast is overridden."""

    async def fake_query(prompt: str) -> str:
        return '{"direction": "down", "confidence": 78, "agreed": false, "reasoning": "bearish divergence", "key_signals": ["divergence"]}'

    hybrid = HybridLLM(
        local_llm=local_forecast_low_confidence,
        confidence_threshold=65.0,
        openai_api_key="sk-test",
    )
    monkeypatch.setattr(hybrid, "_query_remote", fake_query)

    result = await hybrid.analyze_market("EURUSD", "summary")

    assert result["overridden"] is True
    assert result["direction"] == "down"
    assert result["confidence"] == 78.0
    assert result["model_used"] == "gpt-4o-mini"
    assert result["local_forecast"] == "up"


@pytest.mark.asyncio
async def test_hybrid_remote_agrees_keeps_local(local_forecast_low_confidence, monkeypatch) -> None:
    """When the remote model agrees, keep the local forecast."""

    async def fake_query(prompt: str) -> str:
        return '{"direction": "up", "confidence": 70, "agreed": true, "reasoning": "confirms", "key_signals": ["confirm"]}'

    hybrid = HybridLLM(
        local_llm=local_forecast_low_confidence,
        confidence_threshold=65.0,
        openai_api_key="sk-test",
    )
    monkeypatch.setattr(hybrid, "_query_remote", fake_query)

    result = await hybrid.analyze_market("EURUSD", "summary")

    assert result["overridden"] is False
    assert result["direction"] == "up"
