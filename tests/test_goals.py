"""Tests for goal management UI and API."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from borg.db import Database
from borg.web.app import app


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'goals.db'}"
    database = Database(db_url)
    monkeypatch.setattr("borg.web.app.db", database)
    yield database


def test_goals_report_page_renders_create_form(fresh_db) -> None:
    """The Goals & Tasks report page should include the create-goal form."""
    client = TestClient(app)
    response = client.get("/reports/goals-tasks")
    assert response.status_code == 200
    assert "Create Goal" in response.text
    assert 'id="goalForm"' in response.text


def test_dashboard_renders_create_goal_form(fresh_db) -> None:
    """The dashboard should include the create-goal form."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="goalForm"' in response.text
    assert 'id="goalTitle"' in response.text


def test_create_goal_api(fresh_db) -> None:
    """The POST /api/goals endpoint should create a new goal."""
    client = TestClient(app)
    response = client.post(
        "/api/goals",
        data={"title": "Improve forecast calibration", "description": "Test desc", "priority": "90"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["id"], int)

    goals = client.get("/api/goals?status=active").json()
    assert any(g["title"] == "Improve forecast calibration" and g["priority"] == 90 for g in goals)


def test_create_goal_api_defaults_priority(fresh_db) -> None:
    """Creating a goal without priority should default to 50 (server-side)."""
    client = TestClient(app)
    response = client.post(
        "/api/goals",
        data={"title": "Default priority goal"},
    )
    assert response.status_code == 200

    goals = client.get("/api/goals?status=active").json()
    created = next(g for g in goals if g["title"] == "Default priority goal")
    assert created["priority"] == 50
