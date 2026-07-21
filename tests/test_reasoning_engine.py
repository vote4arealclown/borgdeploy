"""Tests for the reasoning engine."""
from __future__ import annotations

from typing import Any

import pytest

from borg.llm import llm
from borg.reasoning import ReasoningEngine


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
        return self._response


@pytest.mark.asyncio
async def test_reasoning_parses_json(monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    fake_llm = _FakeLLM(
        '{"decision": "BUY", "confidence": 75.0, "size_multiplier": 1.2, "reasoning": "looks good", "stop_loss_percent": 1.5, "take_profit_percent": 4.0, "risks": ["volatility"], "overrides_strategy_vote": false}'
    )
    engine = ReasoningEngine(llm_client=fake_llm)
    result = await engine.reason(
        {"symbol": "EURUSD", "candles": [{"close": 1.0}], "features": {}},
        [{"strategy": "mr", "action": "BUY", "confidence": 70.0, "trigger": "x"}],
        "bull_low_vol",
    )
    assert result["decision"] == "BUY"
    assert result["confidence"] == 75.0
    assert result["size_multiplier"] == 1.2


@pytest.mark.asyncio
async def test_reasoning_fallback_on_llm_error(monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    class _BrokenLLM:
        async def generate(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
            raise RuntimeError("boom")

    engine = ReasoningEngine(llm_client=_BrokenLLM())
    result = await engine.reason(
        {"symbol": "EURUSD", "candles": [], "features": {}},
        [{"strategy": "mr", "action": "SELL", "confidence": 80.0, "trigger": "x"}],
        "bear_high_vol",
    )
    assert result["decision"] == "SELL"
    assert "LLM reasoning unavailable" in result["reasoning"]


@pytest.mark.asyncio
async def test_reasoning_accepts_up_down_flat(monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    fake_llm = _FakeLLM('{"decision": "up", "confidence": 60.0}')
    engine = ReasoningEngine(llm_client=fake_llm)
    result = await engine.reason(
        {"symbol": "EURUSD", "candles": [], "features": {}},
        [{"strategy": "mr", "action": "BUY", "confidence": 55.0, "trigger": "x"}],
        "sideways_normal_vol",
    )
    assert result["decision"] == "BUY"
