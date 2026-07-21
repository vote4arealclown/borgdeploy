"""Tests for modular strategies and consensus voting."""
from __future__ import annotations

import pytest

from borg.coordinator import StrategyCoordinator
from borg.db import Database
from borg.llm import llm
from borg.strategies.binary_forecast import BinaryForecastStrategy
from borg.strategies.consensus import ConsensusStrategy
from borg.strategies.mean_reversion import MeanReversionStrategy


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'strategies.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_binary_forecast_strategy(fresh_db, monkeypatch) -> None:
    """Binary forecast strategy should produce a trade when confident."""
    monkeypatch.setattr(llm, "_ollama_available", False)

    strategy = BinaryForecastStrategy("binary", {"confidence_threshold": 60})
    strategy.db = fresh_db
    strategy.llm = llm

    market_data = {
        "symbol": "EURUSD",
        "candles": [
            {"symbol": "EURUSD", "close": 1.0850, "high": 1.0860, "low": 1.0840, "volume": 1_000_000}
        ],
    }
    trades = await strategy.analyze(market_data)

    assert isinstance(trades, list)
    assert market_data.get("last_forecast") is not None
    if trades:
        assert trades[0].symbol == "EURUSD"
        assert trades[0].side in {"buy", "sell"}


@pytest.mark.asyncio
async def test_mean_reversion_strategy(fresh_db) -> None:
    """Mean reversion should produce a trade when price deviates from MA."""
    strategy = MeanReversionStrategy("mr", {"ma_period": 5, "deviation_threshold": 0.01})
    strategy.db = fresh_db

    # Create a steep drop below the moving average.
    candles = [
        {"symbol": "EURUSD", "close": 1.0500, "high": 1.0510, "low": 1.0490, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0800, "high": 1.0810, "low": 1.0790, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0810, "high": 1.0820, "low": 1.0800, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0820, "high": 1.0830, "low": 1.0810, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0830, "high": 1.0840, "low": 1.0820, "volume": 1_000_000},
    ]

    trades = await strategy.analyze({"symbol": "EURUSD", "candles": candles})
    assert len(trades) == 1
    assert trades[0].side == "buy"


@pytest.mark.asyncio
async def test_consensus_requires_two_votes(fresh_db, monkeypatch) -> None:
    """Consensus strategy requires min_votes agreeing strategies."""
    monkeypatch.setattr(llm, "_ollama_available", False)

    binary = BinaryForecastStrategy("binary", {"confidence_threshold": 60})
    binary.db = fresh_db
    binary.llm = llm

    mr = MeanReversionStrategy("mr", {"ma_period": 5, "deviation_threshold": 0.01})
    mr.db = fresh_db

    consensus = ConsensusStrategy("consensus", {"min_votes": 2}, [binary, mr])
    consensus.db = fresh_db
    consensus.llm = llm

    # Case 1: both agree (buy). We craft candles that make MR say buy, and hope
    # the deterministic LLM fallback also says buy. To avoid flakiness, we mock.
    async def mock_analyze(market_data: dict) -> list:
        return [type("T", (), {"symbol": "EURUSD", "side": "buy", "quantity": 1.0, "entry_price": 1.0, "strategy_id": "binary", "confidence": 80})()]

    monkeypatch.setattr(binary, "analyze", mock_analyze)

    candles = [
        {"symbol": "EURUSD", "close": 1.0500, "high": 1.0510, "low": 1.0490, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0800, "high": 1.0810, "low": 1.0790, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0810, "high": 1.0820, "low": 1.0800, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0820, "high": 1.0830, "low": 1.0810, "volume": 1_000_000},
        {"symbol": "EURUSD", "close": 1.0830, "high": 1.0840, "low": 1.0820, "volume": 1_000_000},
    ]
    trades = await consensus.analyze({"symbol": "EURUSD", "candles": candles})
    assert len(trades) == 1
    assert trades[0].side == "buy"
    assert "2 votes" in trades[0].strategy_id


@pytest.mark.asyncio
async def test_coordinator_loads_strategies(fresh_db, monkeypatch) -> None:
    """Strategy coordinator should load strategies from config."""
    monkeypatch.setattr(llm, "_ollama_available", False)

    coordinator = StrategyCoordinator(database=fresh_db, llm=llm)
    names = coordinator.list_strategy_names()

    assert "binary_forecast" in names
    assert "mean_reversion" in names
    assert "consensus_2vote" in names
