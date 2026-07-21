"""Tests for the Pollinations image generation skill."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from borg.modules.image_gen import PollinationsImageClient
from borg.safety import ActionKind, SafetyGate
from borg.schemas import ImageGenerationInput


class _FakeResponse:
    def __init__(self, status_code: int, json_data: Any = None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._json = json_data
        self.content = content

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _AsyncClientMock:
    """Tiny async context manager that records requests and returns a preset response."""

    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_AsyncClientMock":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"method": "GET", "url": url, **kwargs})
        return self.response

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        return self.response


@pytest.fixture
def fresh_db(tmp_path):
    from borg.db import Database

    db_url = f"sqlite:///{tmp_path / 'image_gen_test.db'}"
    return Database(db_url)


@pytest.fixture
def client(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setenv("POLLINATIONS_API_KEY", "sk_test_key")
    monkeypatch.setattr("borg.config.settings.pollinations_enabled", True)
    monkeypatch.setattr("borg.config.settings.pollinations_image_dir", tmp_path / "images")
    gate = SafetyGate(database=fresh_db)
    gate.required.discard(ActionKind.IMAGE_GENERATION)
    return PollinationsImageClient(
        base_url="https://gen.pollinations.ai",
        api_key="sk_test_key",
        image_dir=tmp_path / "images",
        safety_gate=gate,
    )


@pytest.mark.asyncio
async def test_generate_url(client, monkeypatch):
    fake = _AsyncClientMock(_FakeResponse(200))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake)
    result = await client.generate_url("a cat in space", model="flux", width=512, height=512, seed=7)
    assert result.status == "ok"
    assert result.model == "flux"
    assert result.url is not None
    assert "a%20cat%20in%20space" in result.url
    assert "model=flux" in result.url
    assert "width=512" in result.url
    assert "seed=7" in result.url


@pytest.mark.asyncio
async def test_generate_url_safety_blocks(client, monkeypatch):
    client.safety.required.add(ActionKind.IMAGE_GENERATION)
    result = await client.generate_url("a cat in space")
    assert result.status == "needs_confirmation"
    assert "confirmation" in result.error.lower()


@pytest.mark.asyncio
async def test_generate_disabled(client, monkeypatch):
    client.enabled = False
    result = await client.generate_url("a cat in space")
    assert result.status == "disabled"


@pytest.mark.asyncio
async def test_generate_post_saves_image(client, monkeypatch, tmp_path):
    b64_data = "iVBORw0KGgoAAAANSU"  # truncated but valid base64 start, pad below
    b64_data += "A" * (4 - len(b64_data) % 4) if len(b64_data) % 4 else ""
    response_json = {
        "created": 1234567890,
        "data": [{"b64_json": b64_data, "revised_prompt": "a cat"}],
        "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10, "input_tokens_details": {"text_tokens": 10, "image_tokens": 0}},
    }
    fake = _AsyncClientMock(_FakeResponse(200, response_json))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake)
    result = await client.generate("a cat in space", model="flux", width=512, height=512)
    assert result.status == "ok"
    assert result.local_path is not None
    assert Path(result.local_path).exists()


@pytest.mark.asyncio
async def test_generate_post_returns_error(client, monkeypatch):
    fake = _AsyncClientMock(_FakeResponse(500))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake)
    result = await client.generate("a cat in space")
    assert result.status == "error"
    assert result.error is not None


@pytest.mark.asyncio
async def test_list_models(client, monkeypatch):
    fake = _AsyncClientMock(_FakeResponse(200, [{"id": "flux"}, {"id": "zimage"}]))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake)
    models = await client.list_models()
    assert len(models) == 2
    assert models[0]["id"] == "flux"


@pytest.mark.asyncio
async def test_list_models_failure(client, monkeypatch):
    fake = _AsyncClientMock(_FakeResponse(500))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake)
    models = await client.list_models()
    assert models == []


def test_image_generation_schema_validation():
    req = ImageGenerationInput(prompt="a cat", width=1024, height=1024)
    assert req.response_format == "url"
    with pytest.raises(Exception):
        ImageGenerationInput(prompt="a cat", width=64, height=2049)


@pytest.mark.asyncio
async def test_generate_from_schema_url(client, monkeypatch):
    fake = _AsyncClientMock(_FakeResponse(200))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake)
    req = ImageGenerationInput(prompt="a cat", response_format="url")
    result = await client.generate_from_schema(req)
    assert result.status == "ok"
    assert result.url is not None
