"""Smoke tests for the brain loop and memory."""
from __future__ import annotations

import pytest

from borg.brain import Brain, DataFeed
from borg.config import settings
from borg.db import Database
from borg.llm import llm
from borg.memory import Memory
from borg.monitor import monitor


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_data_feed_seed(fresh_db) -> None:
    feed = DataFeed()
    feed.seed(fresh_db)
    for symbol in settings.symbol_list:
        candles = fresh_db.latest_candles(symbol, limit=30)
        assert len(candles) == 30


@pytest.mark.asyncio
async def test_brain_cycle(fresh_db, monkeypatch) -> None:
    # Force deterministic fallback LLM for tests
    monkeypatch.setattr(llm, "_ollama_available", False)

    monkeypatch.setattr(monitor, "should_throttle", lambda: False)
    mem = Memory(database=fresh_db)
    brain = Brain(database=fresh_db, mem=mem)
    await brain.seed()
    result = await brain.cycle()
    assert result["status"] == "ok"
    forecasts = fresh_db.recent_forecasts(limit=10)
    assert len(forecasts) >= len(settings.symbol_list)
