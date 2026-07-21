"""Borg's periodic self-reflection / consciousness thread."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from borg.config import settings
from borg.db import Database, db
from borg.events import EventLog, event_log
from borg.llm import llm
from borg.memory import Memory, memory


class Consciousness:
    """Generates 30-60 second self-summaries from recent events and memory."""

    def __init__(
        self,
        database: Database = db,
        mem: Memory = memory,
        events: EventLog = event_log,
    ) -> None:
        self.db = database
        self.memory = mem
        self.events = events
        self._running = False
        self._last_summary: Optional[datetime] = None

    def _gather_context(self) -> dict[str, Any]:
        recent_events = self.db.recent_events(limit=20)
        recent_forecasts = self.db.recent_forecasts(limit=10)
        recent_learnings = self.db.recent_learnings(limit=5)
        cycles = self.db.last_cycle()
        return {
            "events": recent_events,
            "forecasts": recent_forecasts,
            "learnings": recent_learnings,
            "last_cycle": cycles,
        }

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Recursively make a value JSON-serializable (datetime, date, Decimal)."""
        if isinstance(obj, dict):
            return {k: Consciousness._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Consciousness._json_safe(v) for v in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return obj

    async def summarize(self) -> str:
        context = self._gather_context()
        events_text = "\n".join(
            f"- [{e.get('phase') or 'none'}] {e['message']}" for e in context["events"]
        )
        forecasts_text = "\n".join(
            f"- {f['symbol']} {f['direction'].upper()} {f['confidence']:.1f}%"
            for f in context["forecasts"]
        )
        learnings_text = "\n".join(
            f"- {learning['summary'][:160]}" for learning in context["learnings"]
        )

        prompt = f"""You are Borg's internal narrator. Summarize what Borg has been doing in 2-3 sentences. Be concise and factual; use only the data below.

Recent events:
{events_text or 'None'}

Recent forecasts:
{forecasts_text or 'None'}

Recent learnings:
{learnings_text or 'None'}

Self-summary:"""

        if await llm._check_ollama():
            summary = (await llm.generate(prompt, timeout=settings.consciousness_timeout_seconds)).strip()
        else:
            # Deterministic fallback summary
            event_count = len(context["events"])
            forecast_count = len(context["forecasts"])
            learning_count = len(context["learnings"])
            summary = (
                f"Consciousness snapshot: {event_count} recent events, "
                f"{forecast_count} forecasts, {learning_count} learnings. "
                f"Borg is cycling through {', '.join(settings.symbol_list)}."
            )

        if not summary:
            summary = "Borg is running but has no new reflections to report."

        self.db.insert_conscious_summary(summary, self._json_safe(context))
        await self.memory.observe(summary, kind="summary", source="consciousness")
        self.events.emit(summary, category="consciousness", phase="reflect")
        self._last_summary = datetime.now(timezone.utc)
        return summary

    async def run(self) -> None:
        """Loop forever, summarizing every configured interval."""
        self._running = True
        # Stagger consciousness relative to the brain loop so they don't both
        # hit Ollama at the exact same moment.
        await asyncio.sleep(settings.consciousness_interval_seconds // 2)
        while self._running:
            try:
                await self.summarize()
            except Exception as exc:
                self.events.emit(f"Consciousness error: {exc}", category="system", phase="idle", level="ERROR")
            # Slow loop to avoid piling up LLM requests while Ollama is cycling
            await asyncio.sleep(settings.consciousness_interval_seconds)

    def stop(self) -> None:
        self._running = False

    def last_summary(self) -> Optional[datetime]:
        return self._last_summary


consciousness = Consciousness()
