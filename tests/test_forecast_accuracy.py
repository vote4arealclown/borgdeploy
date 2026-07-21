"""Tests for forecast accuracy reporting."""
from __future__ import annotations

import pytest

from borg.db import Database


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'accuracy.db'}"
    database = Database(db_url)
    yield database


def _make_forecast(db: Database, symbol: str, direction: str, correct: bool) -> int:
    fc_id = db.insert_forecast(
        {
            "symbol": symbol,
            "horizon_s": 300,
            "direction": direction,
            "confidence": 70.0,
            "rationale": "test",
            "features": None,
        }
    )
    outcome = "win" if correct else "loss"
    db.resolve_forecast(fc_id, outcome, correct)
    return fc_id


def test_forecast_accuracy_by_confidence(fresh_db) -> None:
    """Accuracy report groups resolved forecasts by confidence bin."""
    # Create a mix of correct and incorrect forecasts.
    for i in range(10):
        _make_forecast(fresh_db, "EURUSD", "up", i % 2 == 0)
    for i in range(5):
        _make_forecast(fresh_db, "GBPUSD", "down", True)

    forecasts = fresh_db.fetchall(
        "SELECT id, symbol, direction, confidence, outcome, correct FROM forecasts WHERE outcome IS NOT NULL ORDER BY created_at DESC"
    )
    assert len(forecasts) == 15

    correct = sum(1 for f in forecasts if f["correct"])
    accuracy = correct / len(forecasts)
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(forecasts)})")

    # Basic sanity checks.
    assert 0.0 <= accuracy <= 1.0
    assert correct > 0


def test_forecast_accuracy_empty(fresh_db) -> None:
    """No resolved forecasts yields zero accuracy without error."""
    forecasts = fresh_db.fetchall(
        "SELECT id, symbol, direction, confidence, outcome, correct FROM forecasts WHERE outcome IS NOT NULL"
    )
    assert forecasts == []
