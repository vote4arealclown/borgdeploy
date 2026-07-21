"""Tests for the HIP-4 paper-trade dashboard and API."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient

from borg.db import Database
from borg.web.app import app


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'paper_trades.db'}"
    database = Database(db_url)
    monkeypatch.setattr("borg.web.app.db", database)
    yield database


def test_paper_trades_dashboard_page(fresh_db) -> None:
    """The paper-trades HTML dashboard should render."""
    client = TestClient(app)
    response = client.get("/reports/paper-trades")
    assert response.status_code == 200
    assert "HIP-4 Daily Paper Trades" in response.text


def test_paper_trades_api_empty(fresh_db) -> None:
    """The paper-trades API should return an empty list when no trades exist."""
    client = TestClient(app)
    response = client.get("/api/paper_trades")
    assert response.status_code == 200
    assert response.json() == []


def test_paper_trades_api_returns_records(fresh_db) -> None:
    """The paper-trades API should return inserted trades ordered by date."""
    trade = {
        "trade_date": "2026-07-19",
        "underlying": "SOL",
        "direction": "up",
        "side": "YES",
        "target_price": 75.0,
        "entry_price": 75.0,
        "token_price": 0.95,
        "quantity": 1.05263157894737,
        "stake": 1.0,
        "potential_payout": 1.05263157894737,
        "expiry": datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc),
    }
    fresh_db.insert_paper_trade(trade)

    client = TestClient(app)
    response = client.get("/api/paper_trades")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["underlying"] == "SOL"
    assert data[0]["side"] == "YES"
    assert data[0]["outcome"] == "pending"
