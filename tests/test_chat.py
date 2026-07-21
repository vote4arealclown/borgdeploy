"""Smoke tests for chat fallback grounded in live data."""
from __future__ import annotations

import pytest

from borg.chat import ChatEngine
from borg.db import Database


@pytest.fixture
def fresh_db(tmp_path):
    db_url = f"sqlite:///{tmp_path / 'chat_test.db'}"
    database = Database(db_url)
    yield database


@pytest.mark.asyncio
async def test_chat_day_of_week(fresh_db, monkeypatch) -> None:
    engine = ChatEngine(database=fresh_db)
    # Force deterministic fallback
    monkeypatch.setattr("borg.chat.llm._ollama_available", False)
    response = await engine.ask("what day of the week is it")
    assert response["role"] == "assistant"
    assert "day" in response["content"].lower()
    assert response["model_used"] == "fallback_rule_engine"


@pytest.mark.asyncio
async def test_chat_status_when_empty(fresh_db, monkeypatch) -> None:
    engine = ChatEngine(database=fresh_db)
    monkeypatch.setattr("borg.chat.llm._ollama_available", False)
    response = await engine.ask("status")
    assert "borg is running" in response["content"].lower()


@pytest.mark.asyncio
async def test_chat_extract_image_prompt(fresh_db) -> None:
    engine = ChatEngine(database=fresh_db)
    assert engine._extract_image_prompt("draw a cat in space") == "a cat in space"
    assert engine._extract_image_prompt("generate an image of a cyberpunk city") == "a cyberpunk city"
    assert engine._extract_image_prompt("what is the weather") is None
