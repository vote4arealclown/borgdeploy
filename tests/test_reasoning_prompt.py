"""Tests for reasoning prompt construction."""
from __future__ import annotations

from datetime import datetime, timezone

from borg.reasoning import ReasoningPrompt
from borg.schemas import Episode, EpisodeOutcome, EpisodeSignal


def test_prompt_builds_with_context() -> None:
    market_data = {
        "symbol": "EURUSD",
        "candles": [
            {"close": 1.0850, "high": 1.0860, "low": 1.0840, "volume": 1_000_000},
        ],
        "features": {"rsi": 55.0, "volatility": 0.0012},
    }
    signals = [
        {"strategy": "binary_forecast", "action": "BUY", "confidence": 72.0, "trigger": "rsi momentum"},
    ]
    efficacy = {
        "binary_forecast": {"win_rate": 0.65, "sample_size": 20},
    }
    episodes = [
        Episode(
            timestamp=datetime.now(timezone.utc),
            actor="binary_forecast",
            trigger="rsi momentum",
            market_state={},
            regime="bull_low_vol",
            trade_signal=EpisodeSignal(direction="up", confidence=70.0),
            executed=True,
            outcome=EpisodeOutcome(win=True, correct=True, horizon_return=0.001),
        )
    ]
    prompt = ReasoningPrompt.build(market_data, signals, "bull_low_vol", efficacy, episodes)
    assert "EURUSD" in prompt
    assert "binary_forecast" in prompt
    assert "65% wins" in prompt


def test_prompt_handles_empty_context() -> None:
    prompt = ReasoningPrompt.build({"symbol": "X", "candles": [], "features": {}}, [], "unknown", {}, [])
    assert "X" in prompt
    assert "No signals" in prompt
