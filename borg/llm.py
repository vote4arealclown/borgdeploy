"""LLM client with Ollama integration and OpenAI-compatible fallback."""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from typing import Any, Optional

import httpx

from borg.config import settings


def _fake_embedding(text: str, dims: int = 768) -> list[float]:
    """Deterministic hash-based embedding for offline prototyping."""
    rng = random.Random(text)
    vec = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _fake_forecast(symbol: str, market_summary: str) -> dict[str, Any]:
    """Deterministic rule-based forecast used when Ollama is unavailable."""
    summary_lower = market_summary.lower()
    score = 0
    signals: list[str] = []

    if "rsi" in summary_lower:
        try:
            rsi = float(re.search(r"rsi[=: ]+(\d+(?:\.\d+)?)", summary_lower).group(1))  # type: ignore[union-attr]
            if rsi > 70:
                score -= 1
                signals.append("rsi overbought")
            elif rsi < 30:
                score += 1
                signals.append("rsi oversold")
        except Exception:
            pass

    if "volume" in summary_lower and re.search(r"volume.*(high|spike|surge|x normal)", summary_lower):
        score += 1
        signals.append("volume spike")

    if "bullish" in summary_lower or "up" in summary_lower:
        score += 1
    if "bearish" in summary_lower or "down" in summary_lower:
        score -= 1

    rng = random.Random(symbol + market_summary[-120:])
    jitter = rng.uniform(-0.3, 0.3)
    net = score + jitter

    if net > 0.4:
        direction = "up"
        confidence = min(95.0, 65.0 + net * 15.0 + rng.uniform(0, 5))
    elif net < -0.4:
        direction = "down"
        confidence = min(95.0, 65.0 - net * 15.0 + rng.uniform(0, 5))
    else:
        direction = "flat"
        confidence = min(95.0, 50.0 + abs(net) * 20.0)

    return {
        "direction": direction,
        "confidence": round(confidence, 1),
        "key_signals": signals or ["no strong signal"],
        "analysis": f"Fallback rule-based analysis for {symbol}: net score {net:.2f}.",
        "model_used": "fallback_rule_engine",
    }


class LLMClient:
    """Unified LLM interface: Ollama primary, deterministic fallback secondary."""

    def __init__(self) -> None:
        self.ollama_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.embed_model = settings.llm_embed_model
        self.timeout = settings.llm_timeout_seconds
        self._ollama_available: Optional[bool] = None
        # Serialize generation calls so the brain loop and consciousness don't
        # hammer Ollama at the same time on modest hardware.
        self._gen_lock = asyncio.Lock()
        self._fallback_client: Optional[Any] = None

    async def _check_ollama(self) -> bool:
        # An explicitly-set instance flag (e.g. tests) takes precedence.
        if self._ollama_available is not None:
            return self._ollama_available
        if settings.llm_force_fallback:
            self._ollama_available = False
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                self._ollama_available = resp.status_code == 200
        except Exception:
            self._ollama_available = False
        return self._ollama_available or False

    def _openai_fallback(self) -> Optional[httpx.AsyncClient]:
        if settings.llm_fallback_provider != "openai_compat":
            return None
        key = os.environ.get(settings.llm_fallback_api_key_env)
        if not key:
            return None
        return httpx.AsyncClient(
            base_url=settings.llm_fallback_base_url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.timeout,
        )

    async def embed(self, text: str) -> list[float]:
        if not await self._check_ollama():
            return _fake_embedding(text)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                )
                resp.raise_for_status()
                data = resp.json()
                return list(data.get("embedding", [])) or _fake_embedding(text)
        except Exception:
            return _fake_embedding(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    async def generate(self, prompt: str, system: Optional[str] = None, timeout: Optional[float] = None) -> str:
        if not await self._check_ollama():
            return _fake_forecast("UNKNOWN", prompt)["analysis"]
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": settings.llm_temperature, "num_predict": settings.llm_max_tokens},
        }
        if system:
            payload["system"] = system
        request_timeout = timeout if timeout is not None else self.timeout
        try:
            async with self._gen_lock:
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    resp = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                    resp.raise_for_status()
                    return str(resp.json().get("response", ""))
        except Exception as exc:
            return f"Ollama error: {type(exc).__name__}: {exc}"

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Extract the first valid JSON object from a model response.

        Handles markdown fences, trailing prose, and malformed outer text so
        small local models (e.g. qwen2:0.5b) that add chatter still produce a
        usable forecast when possible.
        """
        # Strip common markdown fences and leading/trailing whitespace.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        # Try the whole cleaned string first.
        try:
            return dict(json.loads(cleaned))
        except Exception:
            pass
        # Fall back to the first brace-delimited object.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return dict(json.loads(match.group(0)))
            except Exception:
                pass
        return {}

    async def analyze_market(self, symbol: str, market_summary: str) -> dict[str, Any]:
        if not await self._check_ollama():
            return _fake_forecast(symbol, market_summary)

        prompt = f"""You are a concise market analyst. Given the following market data, respond ONLY with a JSON object.

Market: {symbol}
Summary: {market_summary}

Required JSON format:
{{"direction": "up|down|flat", "confidence": 0-100, "key_signals": ["..."], "analysis": "one sentence rationale"}}
"""
        raw = await self.generate(prompt, timeout=45.0)

        # Network / generation failure already falls back to a deterministic stub.
        fallback = _fake_forecast(symbol, market_summary)
        if raw.startswith("Ollama error:"):
            fallback["model_used"] = f"{self.model}_fallback_rules"
            fallback["raw"] = raw
            return fallback

        parsed = self._parse_json(raw)
        direction = parsed.get("direction", "")
        confidence_raw = parsed.get("confidence")

        # If the model produced unparseable or structurally invalid output, use
        # the deterministic rule engine rather than a meaningless flat default.
        if not parsed or direction not in {"up", "down", "flat"}:
            fallback["model_used"] = f"{self.model}_fallback_rules"
            fallback["raw"] = raw
            return fallback

        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 50.0

        return {
            "direction": direction,
            "confidence": max(0.0, min(100.0, confidence)),
            "key_signals": parsed.get("key_signals", []) or fallback.get("key_signals", []),
            "analysis": parsed.get("analysis", raw[:500]) or fallback.get("analysis", ""),
            "model_used": self.model,
            "raw": raw,
        }

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        """OpenAI-compatible chat completion via Ollama or fallback."""
        if await self._check_ollama():
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.ollama_url}/v1/chat/completions",
                        json={"model": self.model, "messages": messages, "temperature": temperature},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return str(data["choices"][0]["message"]["content"])
            except Exception:
                pass
        # Fallback to deterministic stub
        last = messages[-1]["content"].lower() if messages else ""
        if "forecast" in last:
            return "I can only report forecasts that already exist in Borg's database."
        if "status" in last:
            return "Borg is running. Ask about forecasts, learnings, or recent events."
        return "I'm Borg's assistant. I answer from live system data."


llm = LLMClient()
