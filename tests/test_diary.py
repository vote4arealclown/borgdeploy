"""Tests for the daily diary writer and API."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from borg.db import Database
from borg.modules.diary import DiaryWriter, list_diary_files
from borg.web.app import app


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'diary.db'}"
    database = Database(db_url)
    monkeypatch.setattr("borg.web.app.db", database)
    monkeypatch.setattr("borg.modules.diary.db", database)
    yield database


def test_diary_page_renders(fresh_db) -> None:
    """The diary viewer page should render."""
    client = TestClient(app)
    response = client.get("/diary")
    assert response.status_code == 200
    assert "Daily Diary" in response.text


@pytest.fixture
def diary_output(tmp_path, monkeypatch, fresh_db):
    """Provide an isolated diary output directory and database patched into the singleton writer."""
    output_dir = tmp_path / "diaries"
    output_dir.mkdir()
    from borg.modules import diary as diary_module

    monkeypatch.setattr(diary_module.diary_writer, "output_dir", output_dir)
    monkeypatch.setattr(diary_module.diary_writer, "db", fresh_db)
    yield output_dir


def test_generate_diary_api(fresh_db, diary_output) -> None:
    """The diary generation API should write a Markdown file."""
    client = TestClient(app)
    response = client.post("/api/diary/generate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

    path = Path(data["path"])
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# Borg Daily Diary" in content
    assert date.today().isoformat() in content


def test_list_diaries_api(fresh_db, diary_output) -> None:
    """The diary list API should return generated files."""
    writer = DiaryWriter(database=fresh_db, output_dir=diary_output)
    writer.write_daily_diary(date(2026, 7, 19))
    writer.write_daily_diary(date(2026, 7, 20))

    files = list_diary_files(diary_output)
    assert len(files) == 2
    assert files[0]["date"] == "2026-07-20"
    assert files[1]["date"] == "2026-07-19"


def test_get_diary_api(fresh_db, diary_output) -> None:
    """The diary content API should return the file content."""
    writer = DiaryWriter(database=fresh_db, output_dir=diary_output)
    writer.write_daily_diary(date(2026, 7, 19))

    client = TestClient(app)
    response = client.get("/api/diary/2026-07-19")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "# Borg Daily Diary" in data["content"]


def test_get_missing_diary_api(fresh_db) -> None:
    """The diary content API should return an error for missing files."""
    client = TestClient(app)
    response = client.get("/api/diary/2020-01-01")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"


def test_download_diary_api(fresh_db, diary_output) -> None:
    """The diary download endpoint should return raw Markdown with a content-disposition header."""
    writer = DiaryWriter(database=fresh_db, output_dir=diary_output)
    writer.write_daily_diary(date(2026, 7, 19))

    client = TestClient(app)
    response = client.get("/api/diary/2026-07-19/download")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == "attachment; filename=2026-07-19.md"
    assert "# Borg Daily Diary" in response.text


def test_download_missing_diary_api(fresh_db) -> None:
    """The diary download endpoint should 404 for missing files."""
    client = TestClient(app)
    response = client.get("/api/diary/2020-01-01/download")
    assert response.status_code == 404
