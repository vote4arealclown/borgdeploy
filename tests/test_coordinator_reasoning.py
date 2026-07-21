"""Tests for coordinator reasoning integration."""
from __future__ import annotations

import pytest

from borg.coordinator import StrategyCoordinator
from borg.db import Database
from borg.llm import llm
from borg.memory import Memory


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'reasoning_coord.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_coordinate_returns_reasoning_output(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    coordinator = StrategyCoordinator(database=fresh_db, llm=llm)
    market_data = {
        "symbol": "EURUSD",
        "candles": [
            {"symbol": "EURUSD", "close": 1.0500, "high": 1.0510, "low": 1.0490, "volume": 1_000_000},
            {"symbol": "EURUSD", "close": 1.0800, "high": 1.0810, "low": 1.0790, "volume": 1_000_000},
            {"symbol": "EURUSD", "close": 1.0810, "high": 1.0820, "low": 1.0800, "volume": 1_000_000},
            {"symbol": "EURUSD", "close": 1.0820, "high": 1.0830, "low": 1.0810, "volume": 1_000_000},
            {"symbol": "EURUSD", "close": 1.0830, "high": 1.0840, "low": 1.0820, "volume": 1_000_000},
        ],
        "features": {},
    }
    result = await coordinator.coordinate(market_data, regime="sideways_normal_vol")
    assert "decision" in result
    assert "confidence" in result
    assert "signals_input" in result
    assert result["regime"] == "sideways_normal_vol"


@pytest.mark.asyncio
async def test_coordinate_holds_when_no_signals(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    coordinator = StrategyCoordinator(database=fresh_db, llm=llm)
    # Empty candles -> no mean reversion signal; deterministic fallback likely flat.
    market_data = {"symbol": "EURUSD", "candles": [], "features": {}}
    result = await coordinator.coordinate(market_data, regime="unknown_normal_vol")
    assert result["decision"] == "HOLD"
