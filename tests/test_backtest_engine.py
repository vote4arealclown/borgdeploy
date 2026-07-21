"""Tests for the backtest engine."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from borg.backtest_engine import BacktestEngine
from borg.coordinator import StrategyCoordinator
from borg.db import Database
from borg.llm import llm


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'backtest.db'}"
    database = Database(db_url)
    yield database


def _seed_candles(database: Database, symbol: str, count: int = 50) -> None:
    import random

    rng = random.Random(symbol)
    base = 1.0
    for i in range(count):
        close = base * (1 + rng.gauss(0, 0.001))
        high = close * (1 + abs(rng.gauss(0, 0.0005)))
        low = close * (1 - abs(rng.gauss(0, 0.0005)))
        high, low = max(high, low), min(high, low)
        ts = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
        database.insert_candle(
            {
                "symbol": symbol,
                "ts": ts,
                "open": low + rng.random() * (high - low),
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000,
            }
        )


@pytest.mark.asyncio
async def test_backtest_runs_and_records(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    _seed_candles(fresh_db, "EURUSD")
    coordinator = StrategyCoordinator(database=fresh_db, llm=llm)
    engine = BacktestEngine(coordinator=coordinator, database=fresh_db)
    start = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 2, 0, 0, 0, tzinfo=timezone.utc)
    results = await engine.run_backtest(start, end, symbol="EURUSD", record_episodes=False)
    assert "trade_count" in results
    assert results["symbols"] == ["EURUSD"]


@pytest.mark.asyncio
async def test_backtest_records_episodes(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    from borg.memory import Memory

    _seed_candles(fresh_db, "EURUSD")
    coordinator = StrategyCoordinator(database=fresh_db, llm=llm)
    mem = Memory(database=fresh_db)
    engine = BacktestEngine(coordinator=coordinator, database=fresh_db, memory=mem)
    start = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 2, 0, 0, 0, tzinfo=timezone.utc)
    await engine.run_backtest(start, end, symbol="EURUSD", record_episodes=True)
    episodes = fresh_db.query_episodes(actor="backtest", limit=100)
    assert len(episodes) > 0
