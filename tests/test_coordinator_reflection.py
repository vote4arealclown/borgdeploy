"""Tests for coordinator reflection/efficacy weighting."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from borg.coordinator import StrategyCoordinator
from borg.db import Database
from borg.llm import llm
from borg.memory import Memory
from borg.schemas import Episode, EpisodeOutcome, EpisodeSignal
from borg.strategies.mean_reversion import MeanReversionStrategy


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'coordinator.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_coordinator_reflection_weights_high_efficacy(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    mem = Memory(database=fresh_db)
    for i in range(10):
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

    coordinator = StrategyCoordinator(database=fresh_db, llm=llm, mem=mem)
    candles = [
        {"symbol": "EURUSD", "close": 1.0500, "high": 1.0510, "low": 1.0490, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0800, "high": 1.0810, "low": 1.0790, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0810, "high": 1.0820, "low": 1.0800, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0820, "high": 1.0830, "low": 1.0810, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0830, "high": 1.0840, "low": 1.0820, "volume": 1_000_000},
    ]
    market_data = {"symbol": "EURUSD", "candles": candles, "features": {}}

    trades = await coordinator.execute_with_reflection(market_data, regime="sideways_normal_vol")
    assert len(trades) == 1
    assert trades[0].side == "buy"
    assert trades[0].confidence > 50.0


@pytest.mark.asyncio
async def test_coordinator_weight_roundtrip(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    coordinator = StrategyCoordinator(database=fresh_db, llm=llm)
    weights = {("mr", "bull_low_vol"): 1.5}
    coordinator.apply_weights(weights)
    assert coordinator.get_weights() == weights
    coordinator.reset_weights()
    assert coordinator.get_weights() == {}
