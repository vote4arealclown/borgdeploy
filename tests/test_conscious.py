"""Tests for the consciousness / self-reflection module."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from borg.conscious import Consciousness
from borg.db import Database


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    database = Database(db_url)
    yield database


def test_json_safe_converts_datetime_and_decimal() -> None:
    context = {
        "events": [
            {
                "id": 1,
                "created_at": datetime(2026, 7, 18, 8, 40, 25, tzinfo=timezone.utc),
                "message": "test",
            }
        ],
        "forecasts": [
            {"symbol": "EURUSD", "confidence": Decimal("0.85")},
        ],
        "meta": {"date": date(2026, 7, 18)},
    }
    safe = Consciousness._json_safe(context)

    assert safe["events"][0]["created_at"] == "2026-07-18T08:40:25+00:00"
    assert safe["forecasts"][0]["confidence"] == 0.85
    assert safe["meta"]["date"] == "2026-07-18"


@pytest.mark.asyncio
async def test_summarize_stores_json_safe_context(fresh_db, monkeypatch) -> None:
    from borg.llm import llm

    async def _offline():
        return False

    monkeypatch.setattr(llm, "_check_ollama", _offline)

    # Seed a cycle so last_cycle returns a row with datetime fields
    cycle_id = fresh_db.start_cycle()
    fresh_db.finish_cycle(cycle_id, "ok", "ok")

    conscious = Consciousness(database=fresh_db)
    summary = await conscious.summarize()

    assert summary
    # The inserted row should deserialize cleanly
    rows = fresh_db.fetchall("SELECT context FROM conscious_summaries ORDER BY id DESC LIMIT 1")
    assert len(rows) == 1
    assert rows[0]["context"] is not None
