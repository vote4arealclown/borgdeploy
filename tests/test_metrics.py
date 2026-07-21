"""Tests for Prometheus metrics endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient

from borg.web.app import app


def test_metrics_endpoint() -> None:
    """The /metrics endpoint should return Prometheus exposition format."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "borg_brain_cycles_total" in body
    assert "borg_forecasts_generated_total" in body
    assert "borg_errors_total" in body
