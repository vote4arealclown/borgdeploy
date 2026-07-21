"""Tests for the saskpoly.xyz-style report marketplace."""
from __future__ import annotations

from datetime import date

import pytest

from borg.db import Database
from borg.modules.reports import ReportEngine


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'reports.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_generate_daily_report(fresh_db) -> None:
    engine = ReportEngine(database=fresh_db)
    report = engine.generate_daily_report(date(2026, 7, 17))

    assert report["category"] == "daily"
    assert report["slug"] == "daily-report-2026-07-17"
    assert fresh_db.get_report(report["slug"]) is not None
    deltas = fresh_db.get_market_deltas("2026-07-17")
    assert len(deltas) > 0


@pytest.mark.asyncio
async def test_generate_all_reports(fresh_db) -> None:
    engine = ReportEngine(database=fresh_db)
    reports = engine.generate_all(date(2026, 7, 17))

    slugs = {r["slug"] for r in reports}
    assert "daily-report-2026-07-17" in slugs
    assert "hr-report-2026-07-17" in slugs
    assert "brent-report-2026-07-17" in slugs
    assert "coffee-news-2026-07-17" in slugs


@pytest.mark.asyncio
async def test_pdf_generation(fresh_db) -> None:
    engine = ReportEngine(database=fresh_db)
    engine.generate_daily_report(date(2026, 7, 17))
    pdf = engine.generate_pdf("daily-report-2026-07-17")

    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")


@pytest.mark.asyncio
async def test_seed_scheduled_events(fresh_db) -> None:
    engine = ReportEngine(database=fresh_db)
    engine.seed_sample_events()
    events = fresh_db.list_scheduled_events()
    assert len(events) >= 1
