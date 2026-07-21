"""Tests for the reflection engine."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from borg.db import Database
from borg.llm import llm
from borg.memory import Memory
from borg.reflection import ReflectionEngine
from borg.schemas import Episode, EpisodeOutcome, EpisodeSignal


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'reflection.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_reflect_no_history(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    engine = ReflectionEngine(mem)
    result = await engine.reflect_on_signal(
        {"strategy": "binary_forecast", "trigger": "rsi=70", "confidence": 80.0},
        "bull_low_vol",
    )
    assert result["confidence_adj"] == 1.0
    assert result["sample_size"] == 0
    assert "No historical precedent" in result["reflection"]


@pytest.mark.asyncio
async def test_reflect_with_history(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    for i in range(10):
        await mem.store_episode(
            Episode(
                timestamp=datetime.now(timezone.utc),
                actor="binary_forecast",
                trigger="rsi=70",
                market_state={"latest_close": 1.0},
                regime="bull_low_vol",
                trade_signal=EpisodeSignal(direction="up", confidence=80.0),
                executed=True,
                outcome=EpisodeOutcome(win=i < 8, correct=i < 8, horizon_return=0.001 if i < 8 else -0.002),
            )
        )

    engine = ReflectionEngine(mem)
    result = await engine.reflect_on_signal(
        {"strategy": "binary_forecast", "trigger": "rsi=70", "confidence": 80.0},
        "bull_low_vol",
    )
    assert result["sample_size"] == 10
    assert result["win_rate_in_regime"] == 0.8
    assert result["confidence_adj"] > 1.0


@pytest.mark.asyncio
async def test_periodic_reflection(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    await mem.store_episode(
        Episode(
            timestamp=datetime.now(timezone.utc),
            actor="mr",
            trigger="deviation",
            market_state={"latest_close": 1.0},
            regime="sideways_normal_vol",
            trade_signal=EpisodeSignal(direction="buy", confidence=70.0),
            executed=True,
            outcome=EpisodeOutcome(win=True, correct=True, horizon_return=0.001),
        )
    )

    engine = ReflectionEngine(mem)
    summary = await engine.periodic_reflection(window_hours=24)
    assert "mr::sideways_normal_vol" in summary["reflections"]
    assert summary["reflections"]["mr::sideways_normal_vol"]["wins"] == 1
