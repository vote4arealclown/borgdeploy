"""Tests for reasoning audit and calibration."""
from __future__ import annotations

import pytest

from borg.db import Database
from borg.reasoning_audit import ReasoningAudit


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'reasoning_audit.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_record_and_query_audit(fresh_db) -> None:
    audit = ReasoningAudit(database=fresh_db)
    audit_id = await audit.record(
        reasoning_output={"decision": "BUY", "confidence": 70.0, "reasoning": "test"},
        forecast_outcome={"win": True, "correct": True, "horizon_return": 0.001},
        forecast_id=42,
    )
    assert audit_id > 0
    rows = fresh_db.query_reasoning_audits(limit=10)
    assert len(rows) == 1
    assert rows[0]["forecast_id"] == 42
    assert rows[0]["outcome_win"] is True


@pytest.mark.asyncio
async def test_calibration_report(fresh_db) -> None:
    audit = ReasoningAudit(database=fresh_db)
    for i in range(10):
        confidence = 75.0 if i < 5 else 25.0
        win = i < 7
        await audit.record(
            reasoning_output={"decision": "BUY", "confidence": confidence, "reasoning": "x"},
            forecast_outcome={"win": win, "correct": win, "horizon_return": 0.001 if win else -0.001},
        )

    report = await audit.get_calibration_report(window_days=30)
    assert report["total_audits"] == 10
    assert "75.0" in report["buckets"] or "25.0" in report["buckets"]
    assert report["mean_calibration_error"] >= 0.0
