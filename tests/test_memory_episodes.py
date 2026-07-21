"""Tests for episode storage, retrieval, and efficacy calculation."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from borg.db import Database
from borg.llm import llm
from borg.memory import Memory
from borg.schemas import Episode, EpisodeOutcome, EpisodeSignal


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'episodes.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_store_and_query_episode(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    episode = Episode(
        timestamp=datetime.now(timezone.utc),
        actor="binary_forecast",
        trigger="rsi=65 vol=0.0020",
        market_state={"latest_close": 1.085, "features": {"rsi": 65.0}},
        regime="bull_low_vol",
        trade_signal=EpisodeSignal(direction="up", confidence=75.0),
        executed=True,
        outcome=EpisodeOutcome(win=True, correct=True, horizon_return=0.001),
    )
    episode_id = await mem.store_episode(episode)
    assert episode_id > 0

    rows = fresh_db.query_episodes(actor="binary_forecast", regime="bull_low_vol")
    assert len(rows) == 1
    assert rows[0]["actor"] == "binary_forecast"
    assert rows[0]["outcome"]["win"] is True


@pytest.mark.asyncio
async def test_find_similar_episodes(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    for i in range(5):
        await mem.store_episode(
            Episode(
                timestamp=datetime.now(timezone.utc),
                actor="mean_reversion",
                trigger="deviation=-0.02",
                market_state={"latest_close": 1.0 - i * 0.01},
                regime="bear_high_vol",
                trade_signal=EpisodeSignal(direction="buy", confidence=60.0),
                executed=True,
                outcome=EpisodeOutcome(win=i % 2 == 0, correct=i % 2 == 0, horizon_return=0.0001 * i),
            )
        )

    similar = await mem.find_similar_episodes("bear_high_vol", "deviation=-0.02", limit=3)
    assert len(similar) == 3


@pytest.mark.asyncio
async def test_strategy_efficacy(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    for i in range(10):
        await mem.store_episode(
            Episode(
                timestamp=datetime.now(timezone.utc),
                actor="test_strategy",
                trigger="signal",
                market_state={"latest_close": 1.0},
                regime="sideways_normal_vol",
                trade_signal=EpisodeSignal(direction="up", confidence=70.0),
                executed=True,
                outcome=EpisodeOutcome(
                    win=i < 7,
                    correct=i < 7,
                    horizon_return=0.001 if i < 7 else -0.002,
                ),
            )
        )

    efficacy = await mem.get_strategy_efficacy("test_strategy", "sideways_normal_vol", window_days=30)
    assert efficacy.strategy == "test_strategy"
    assert efficacy.regime == "sideways_normal_vol"
    assert efficacy.win_rate == 0.7
    assert efficacy.sample_size == 10
