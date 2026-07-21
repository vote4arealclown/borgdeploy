"""Hybrid LLM strategy: fast local model + premium fallback for low confidence."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import httpx

from borg.config import settings
from borg.llm import LLMClient
from borg.metrics import llm_inference_latency_seconds, timed


class HybridLLM:
    """Two-tier inference: fast local Ollama + optional OpenAI-compatible fallback.

    When the local model's confidence is below the threshold and an API key is
    configured, the forecast is escalated to a more capable remote model.
    """

    def __init__(
        self,
        local_llm: Optional[LLMClient] = None,
        confidence_threshold: float = 65.0,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: Optional[str] = None,
    ) -> None:
        self.local = local_llm or LLMClient()
        self.confidence_threshold = confidence_threshold
        self.openai_api_key = openai_api_key or os.environ.get(settings.llm_fallback_api_key_env)
        self.openai_base_url = openai_base_url or settings.llm_fallback_base_url or "https://api.openai.com/v1"
        self.openai_model = openai_model or settings.llm_fallback_model or "gpt-4o-mini"
        self.stats: dict[str, int] = {"local": 0, "remote": 0}

    def _remote_available(self) -> bool:
        return bool(self.openai_api_key and self.openai_base_url)

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Extract the first JSON object from a text response."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}

    async def _query_remote(self, prompt: str) -> str:
        """Query the OpenAI-compatible fallback endpoint."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.openai_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                json={
                    "model": self.openai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])

    async def analyze_market(self, symbol: str, market_summary: str) -> dict[str, Any]:
        """Generate a forecast, escalating to the remote model when uncertain."""
        with timed(llm_inference_latency_seconds):
            local = await self.local.analyze_market(symbol, market_summary)

        self.stats["local"] += 1
        confidence = float(local.get("confidence", 0.0))
        direction = local.get("direction", "flat")

        if confidence >= self.confidence_threshold or not self._remote_available():
            return {
                "direction": direction,
                "confidence": confidence,
                "key_signals": local.get("key_signals", []),
                "analysis": local.get("analysis", ""),
                "model_used": local.get("model_used", "unknown"),
                "overridden": False,
            }

        # Escalate to remote model for a second opinion.
        prompt = f"""You are a precise market analyst. A junior model produced this forecast:

Symbol: {symbol}
Market summary: {market_summary}
Junior forecast: {direction} (confidence {confidence:.1f}%)

Do you agree? Provide a refined forecast.
Respond ONLY with JSON:
{{"direction": "up|down|flat", "confidence": 0-100, "agreed": true|false, "reasoning": "...", "key_signals": ["..."]}}
"""
        try:
            with timed(llm_inference_latency_seconds):
                raw = await self._query_remote(prompt)
            remote = self._parse_json(raw)
            self.stats["remote"] += 1

            if remote.get("agreed", False):
                return {
                    "direction": direction,
                    "confidence": confidence,
                    "key_signals": local.get("key_signals", []),
                    "analysis": local.get("analysis", ""),
                    "model_used": local.get("model_used", "unknown"),
                    "overridden": False,
                    "remote_reasoning": remote.get("reasoning", ""),
                }

            remote_confidence = max(0.0, min(100.0, float(remote.get("confidence", 50.0))))
            remote_direction = remote.get("direction", "flat")
            if remote_direction not in {"up", "down", "flat"}:
                remote_direction = "flat"

            return {
                "direction": remote_direction,
                "confidence": remote_confidence,
                "key_signals": remote.get("key_signals", []),
                "analysis": remote.get("reasoning", ""),
                "model_used": self.openai_model,
                "overridden": True,
                "override_reason": remote.get("reasoning", ""),
                "local_forecast": direction,
                "local_confidence": confidence,
            }
        except Exception as exc:
            # Remote failed; fall back to local result.
            return {
                "direction": direction,
                "confidence": confidence,
                "key_signals": local.get("key_signals", []),
                "analysis": local.get("analysis", ""),
                "model_used": local.get("model_used", "unknown"),
                "overridden": False,
                "remote_error": str(exc),
            }

    async def embed(self, text: str) -> list[float]:
        """Delegate embedding to the local LLM client."""
        return await self.local.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Delegate batch embedding to the local LLM client."""
        return await self.local.embed_batch(texts)

    async def generate(self, prompt: str, system: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """Delegate generation to the local LLM client."""
        return await self.local.generate(prompt, system=system, timeout=timeout)

    async def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        """Delegate chat to the local LLM client."""
        return await self.local.chat(messages, temperature=temperature)

    def stats_summary(self) -> dict[str, Any]:
        """Return usage statistics for local vs remote inference."""
        total = self.stats["local"] + self.stats["remote"]
        remote_pct = round(100 * self.stats["remote"] / total, 1) if total else 0.0
        return {
            "local": self.stats["local"],
            "remote": self.stats["remote"],
            "total": total,
            "remote_pct": remote_pct,
        }


hybrid_llm = HybridLLM()


def should_use_hybrid() -> bool:
    """Return True if a remote fallback is configured and available."""
    return bool(
        settings.llm_fallback_provider == "openai_compat"
        and hybrid_llm.openai_api_key
        and hybrid_llm.openai_base_url
    )
