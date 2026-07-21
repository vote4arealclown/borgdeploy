"""Tests for the self-analysis engine."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from borg.db import Database
from borg.llm import llm
from borg.memory import Memory
from borg.schemas import Episode, EpisodeOutcome, EpisodeSignal
from borg.self_analysis import SelfAnalysisEngine


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'self_analysis.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_analyze_performance(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    for i in range(10):
        await mem.store_episode(
            Episode(
                timestamp=datetime.now(timezone.utc),
                actor="mr",
                trigger="x",
                market_state={},
                regime="bull_low_vol",
                trade_signal=EpisodeSignal(direction="up", confidence=70.0),
                executed=True,
                outcome=EpisodeOutcome(win=i < 7, correct=i < 7, horizon_return=0.001 if i < 7 else -0.001),
            )
        )

    engine = SelfAnalysisEngine(database=fresh_db)
    analysis = engine.analyze_performance("daily")
    assert analysis["trades_count"] == 10
    assert analysis["win_rate"] == 0.7
    assert "mr" in analysis["by_strategy"]
    assert "bull_low_vol" in analysis["by_regime"]


@pytest.mark.asyncio
async def test_identify_insights(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    for _ in range(5):
        await mem.store_episode(
            Episode(
                timestamp=datetime.now(timezone.utc),
                actor="mr",
                trigger="x",
                market_state={},
                regime="bull_low_vol",
                trade_signal=EpisodeSignal(direction="up", confidence=70.0),
                executed=True,
                outcome=EpisodeOutcome(win=True, correct=True, horizon_return=0.001),
            )
        )

    engine = SelfAnalysisEngine(database=fresh_db)
    analysis = engine.analyze_performance("daily")
    insights = engine.identify_insights(analysis)
    assert any("Top performer" in i for i in insights)


def test_surface_risks() -> None:
    engine = SelfAnalysisEngine(database=None)  # type: ignore[arg-type]
    risks = engine.surface_risks({"max_dd": 0.06, "win_rate": 0.4, "sharpe": 0.8, "trades_count": 10})
    assert any("Drawdown" in r for r in risks)
    assert any("Win rate" in r for r in risks)
