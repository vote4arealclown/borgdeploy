"""Tests for the consciousness reporter."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from borg.consciousness_reporter import ConsciousnessReporter
from borg.db import Database
from borg.llm import llm
from borg.memory import Memory
from borg.schemas import Episode, EpisodeOutcome, EpisodeSignal
from borg.self_analysis import SelfAnalysisEngine


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'consciousness.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_generate_daily_report(fresh_db, monkeypatch) -> None:
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

    analysis = SelfAnalysisEngine(database=fresh_db)
    reporter = ConsciousnessReporter(analysis, llm_client=llm)
    report = await reporter.generate_daily_report()
    assert "BORG Daily Consciousness Report" in report
    assert "5" in report


@pytest.mark.asyncio
async def test_generate_daily_report_no_trades(fresh_db, monkeypatch) -> None:
    monkeypatch.setattr(llm, "_ollama_available", False)

    analysis = SelfAnalysisEngine(database=fresh_db)
    reporter = ConsciousnessReporter(analysis, llm_client=llm)
    report = await reporter.generate_daily_report()
    assert "No trades" in report


@pytest.mark.asyncio
async def test_generate_weekly_report(fresh_db, monkeypatch) -> None:
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

    analysis = SelfAnalysisEngine(database=fresh_db)
    reporter = ConsciousnessReporter(analysis, llm_client=llm)
    report = await reporter.generate_weekly_report()
    assert "BORG Weekly Consciousness Report" in report
