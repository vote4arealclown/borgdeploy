"""Tests for the LLM client fallback and JSON parsing behaviour."""
from __future__ import annotations

import pytest

from borg.llm import LLMClient, _fake_forecast


@pytest.mark.asyncio
async def test_llm_ollama_offline_uses_rule_fallback(monkeypatch) -> None:
    """When Ollama is unreachable, analyze_market uses the deterministic rule engine."""
    client = LLMClient()
    monkeypatch.setattr(client, "_ollama_available", False)

    result = await client.analyze_market("BTC", "rsi: 28, volume spike, bullish")

    assert result["direction"] in {"up", "down", "flat"}
    assert 0 <= result["confidence"] <= 100
    assert "fallback" in result.get("model_used", "")


@pytest.mark.asyncio
async def test_llm_malformed_json_falls_back_to_rules(monkeypatch) -> None:
    """When Ollama returns malformed JSON, the deterministic rule engine is used."""
    client = LLMClient()
    monkeypatch.setattr(client, "_ollama_available", True)

    async def fake_generate(prompt: str, system=None, timeout=None):
        return "This is not JSON at all."

    monkeypatch.setattr(client, "generate", fake_generate)

    result = await client.analyze_market("ETH", "rsi: 72, bearish")

    assert result["direction"] in {"up", "down", "flat"}
    assert "fallback_rules" in result.get("model_used", "")
    assert "raw" in result


@pytest.mark.asyncio
async def test_llm_markdown_fenced_json_is_parsed(monkeypatch) -> None:
    """JSON wrapped in markdown fences should be extracted and used."""
    client = LLMClient()
    monkeypatch.setattr(client, "_ollama_available", True)

    async def fake_generate(prompt: str, system=None, timeout=None):
        return '```json\n{"direction": "up", "confidence": 78.5, "key_signals": ["breakout"], "analysis": "strong"}\n```'

    monkeypatch.setattr(client, "generate", fake_generate)

    result = await client.analyze_market("SOL", "breakout")

    assert result["direction"] == "up"
    assert result["confidence"] == 78.5
    assert "breakout" in result["key_signals"]
    assert result["model_used"] == client.model


@pytest.mark.asyncio
async def test_llm_invalid_direction_falls_back(monkeypatch) -> None:
    """A JSON object with an invalid direction value should trigger the rule fallback."""
    client = LLMClient()
    monkeypatch.setattr(client, "_ollama_available", True)

    async def fake_generate(prompt: str, system=None, timeout=None):
        return '{"direction": "sideways", "confidence": 80}'

    monkeypatch.setattr(client, "generate", fake_generate)

    result = await client.analyze_market("XRP", "rsi: 45")

    assert result["direction"] in {"up", "down", "flat"}
    assert "fallback_rules" in result.get("model_used", "")


def test_fake_forecast_extracts_rsi() -> None:
    """The deterministic rule engine should extract RSI from the summary."""
    result = _fake_forecast("BTC", "rsi: 25, oversold, volume spike")
    assert result["direction"] == "up"
    assert any("rsi oversold" in s for s in result["key_signals"])
