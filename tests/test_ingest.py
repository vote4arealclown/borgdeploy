"""Smoke tests for CSV/JSON ingestion."""
from __future__ import annotations

import pytest

from borg.ingest import IngestWatcher


@pytest.fixture
def fresh_db(tmp_path):
    from borg.db import Database

    db_url = f"sqlite:///{tmp_path / 'ingest_test.db'}"
    database = Database(db_url)
    yield database


@pytest.fixture
def watcher(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setattr("borg.config.settings.input_path", str(tmp_path / "input"))
    monkeypatch.setattr("borg.config.settings.output_path", str(tmp_path / "output"))
    w = IngestWatcher(database=fresh_db)
    w.input_paths = [tmp_path / "input"]
    w.processed_dir = tmp_path / "output" / "processed"
    return w


@pytest.mark.asyncio
async def test_ingest_csv(watcher, tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    csv_file = input_dir / "candles.csv"
    csv_file.write_text(
        "symbol,ts,open,high,low,close,volume\n"
        "EURUSD,2025-01-15T12:30:00Z,1.0840,1.0860,1.0820,1.0850,1500000\n"
        "GBPUSD,2025-01-15T12:31:00Z,1.2700,1.2720,1.2680,1.2710,1200000\n"
    )
    results = await watcher.scan()
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert results[0]["inserted"] == 2
    assert results[0]["errors"] == []
    assert not csv_file.exists()


@pytest.mark.asyncio
async def test_ingest_json(watcher, tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    json_file = input_dir / "candles.json"
    json_file.write_text(
        '[{"symbol": "USDJPY", "ts": "2025-01-15T12:32:00Z", '
        '"open": 151.5, "high": 151.8, "low": 151.2, "close": 151.6, "volume": 1000000}]'
    )
    results = await watcher.scan()
    assert len(results) == 1
    assert results[0]["inserted"] == 1
