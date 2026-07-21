"""Chat interface grounded in live Borg data."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from borg.config import settings
from borg.db import Database, db
from borg.events import EventLog, event_log
from borg.llm import llm
from borg.modules.image_gen import image_client
from borg.monitor import monitor


class ChatEngine:
    """Answer user questions using actual Borg state and history."""

    def __init__(
        self,
        database: Database = db,
        events: EventLog = event_log,
    ) -> None:
        self.db = database
        self.events = events

    @staticmethod
    def _ts(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.strftime("%H:%M:%S")
        s = str(value)
        return s[11:19] if len(s) >= 19 else s

    def _gather_context(self) -> dict[str, Any]:
        status = monitor.status()
        last_cycle = self.db.last_cycle()
        now = datetime.now(timezone.utc)
        status_dict = status.model_dump()

        raw_forecasts = self.db.recent_forecasts(limit=5)
        forecasts_summary = []
        for f in raw_forecasts:
            ts = self._ts(f.get("created_at"))
            forecasts_summary.append(
                f"- {f['symbol']} {f['direction'].upper()} {float(f['confidence']):.1f}% at {ts}"
            )

        raw_learnings = self.db.recent_learnings(limit=3)
        learnings_summary = [f"- {learning['summary'][:120]}" for learning in raw_learnings]

        raw_events = self.db.recent_events(limit=6)
        events_summary = []
        for e in raw_events:
            ts = self._ts(e.get("ts"))
            phase = f"[{e.get('phase') or 'none'}] " if e.get("phase") else ""
            events_summary.append(f"- {ts} {phase}{e['message'][:100]}")

        last_cycle_ts = self._ts(last_cycle.get("started_at")) if last_cycle else None

        return {
            "current_time": now.isoformat(),
            "current_day": now.strftime("%A"),
            "current_date": now.strftime("%Y-%m-%d"),
            "status": {
                "cpu_percent": status_dict["cpu_percent"],
                "memory_used_mb": round(status_dict["memory_used_mb"], 1),
                "memory_total_mb": round(status_dict["memory_total_mb"], 1),
                "active_symbols": list(settings.symbol_list),
                "last_cycle": last_cycle_ts,
            },
            "recent_forecasts": forecasts_summary,
            "recent_learnings": learnings_summary,
            "recent_events": events_summary,
        }

    @staticmethod
    def _extract_image_prompt(user_message: str) -> str | None:
        """Detect image-generation intent and return the prompt, if any."""
        msg = user_message.lower().strip()
        prefixes = (
            "generate an image of ",
            "generate image of ",
            "generate a picture of ",
            "make an image of ",
            "make a picture of ",
            "create an image of ",
            "create a picture of ",
            "draw ",
            "draw an image of ",
            "draw a picture of ",
            "image of ",
            "picture of ",
        )
        for prefix in prefixes:
            if msg.startswith(prefix):
                return user_message[len(prefix):].strip()
        # Handle "... for me" / "... please" suffixes
        for phrase in ("create an image", "generate an image", "make an image", "draw an image"):
            if phrase in msg:
                start = msg.index(phrase) + len(phrase)
                remainder = user_message[start:].strip()
                for suffix in ("of ", "for me", "please", ", please", "."):
                    if remainder.lower().startswith(suffix):
                        remainder = remainder[len(suffix):].strip()
                if remainder:
                    return remainder
                # Fallback: use everything before the trigger phrase as context
                return user_message[:msg.index(phrase)].strip()
        return None

    async def _try_image_generation(self, user_message: str) -> dict[str, Any] | None:
        """If the user asks for an image, generate it and return a chat response."""
        prompt = self._extract_image_prompt(user_message)
        if not prompt:
            return None
        result = await image_client.generate_url(prompt=prompt)
        if result.status == "ok" and result.url:
            answer = f"Here is your image: {result.url}\n\nModel: {result.model}"
            self.db.add_message("assistant", answer, metadata={"model": result.model, "image_url": result.url})
            self.events.emit(f"Chat generated image: {prompt[:80]}", category="chat", metadata={"url": result.url})
            return {"role": "assistant", "content": answer, "model_used": result.model, "image_url": result.url}
        if result.status == "needs_confirmation":
            answer = (
                "Image generation requires approval. "
                "Visit /image-gen and click Approve, or ask an admin to remove 'image_generation' from safety.require_confirmation_for."
            )
            self.db.add_message("assistant", answer, metadata={"model": "safety_gate"})
            return {"role": "assistant", "content": answer, "model_used": "safety_gate"}
        answer = f"Could not generate image: {result.error or result.status}"
        self.db.add_message("assistant", answer, metadata={"model": "image_gen_error"})
        return {"role": "assistant", "content": answer, "model_used": "image_gen_error"}

    def _fallback_answer(self, user_message: str, context: dict[str, Any]) -> str:
        """Rule-based answers when Ollama is unavailable."""
        msg = user_message.lower()
        forecasts = context["recent_forecasts"]
        learnings = context["recent_learnings"]
        events = context["recent_events"]
        status = context["status"]
        current_day = context["current_day"]
        current_date = context["current_date"]
        current_time = context["current_time"]

        # Date / time awareness
        if any(k in msg for k in ("day of the week", "day is it", "what day", "today")):
            return f"Today is {current_day}, {current_date}."

        if any(k in msg for k in ("time is it", "current time", "what time")):
            return f"Current UTC time is {current_time}."

        if any(k in msg for k in ("status", "health", "how are you", "cpu", "memory")):
            return (
                f"Borg is running. CPU: {status['cpu_percent']:.1f}%, "
                f"Memory: {status['memory_used_mb']:.0f}/{status['memory_total_mb']:.0f} MB. "
                f"Last cycle: {status['last_cycle'] or 'never'}. "
                f"Watching: {', '.join(status['active_symbols'])}."
            )

        if any(k in msg for k in ("forecast", "prediction", "trade", "signal")):
            if not forecasts:
                return "No forecasts have been generated yet."
            return "Latest forecasts:\n" + "\n".join(forecasts[:5])

        if any(k in msg for k in ("learning", "memory", "remember", "lesson")):
            if not learnings:
                return "No learnings stored yet."
            return "Recent learnings:\n" + "\n".join(learnings[:5])

        if any(k in msg for k in ("event", "log", "doing", "work", "happen")):
            if not events:
                return "No events logged yet."
            return "Recent events:\n" + "\n".join(events[:8])

        if any(k in msg for k in ("hello", "hi ", "hey", "greetings")):
            return (
                f"Hello. I'm Borg's onboard assistant. Today is {current_day}, {current_date}. "
                "Ask me about system status, latest forecasts, recent learnings, events, or the current time."
            )

        return (
            "I'm Borg's onboard assistant. Today is "
            f"{current_day}, {current_date}. Ask me about: status, latest forecasts, "
            "recent learnings, what Borg is working on, or the day/time."
        )

    async def ask(self, user_message: str) -> dict[str, Any]:
        self.db.add_message("user", user_message)

        image_response = await self._try_image_generation(user_message)
        if image_response:
            return image_response

        context = self._gather_context()

        if not await llm._check_ollama():
            answer = self._fallback_answer(user_message, context)
            self.db.add_message("assistant", answer, metadata={"model": "fallback_rule_engine"})
            self.events.emit(f"Chat: {user_message}", category="chat", metadata={"response": answer[:200]})
            return {"role": "assistant", "content": answer, "model_used": "fallback_rule_engine"}

        forecasts_text = "\n".join(context["recent_forecasts"]) or "None"
        learnings_text = "\n".join(context["recent_learnings"]) or "None"
        events_text = "\n".join(context["recent_events"]) or "None"
        prompt = f"""You are Borg, a concise AI agent dashboard assistant. Use the live data below to answer the user's question. Do not invent facts about forecasts or system state. For simple questions like "what day is it?" or greetings, answer naturally using the current time provided.

Current time:
- Day: {context['current_day']}
- Date: {context['current_date']}
- UTC: {context['current_time']}

System status:
- CPU: {context['status']['cpu_percent']:.1f}%
- Memory: {context['status']['memory_used_mb']:.0f}/{context['status']['memory_total_mb']:.0f} MB
- Symbols watched: {', '.join(context['status']['active_symbols'])}
- Last cycle: {context['status']['last_cycle'] or 'never'}

Recent forecasts:
{forecasts_text}

Recent learnings:
{learnings_text}

Recent events:
{events_text}

User question: {user_message}

Answer in 2-4 sentences. Mention specific symbols and numbers when relevant.
"""
        raw = await llm.generate(prompt, timeout=60.0)
        if raw.startswith("Ollama error:"):
            raw = await llm.generate(prompt, timeout=60.0)
        answer = raw.strip() or "I'm not sure how to answer that from the current data."
        self.db.add_message("assistant", answer, metadata={"model": settings.llm_model})
        self.events.emit(f"Chat: {user_message}", category="chat", metadata={"response": answer[:200]})
        return {"role": "assistant", "content": answer, "model_used": settings.llm_model}

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.recent_messages(limit=limit)
        rows.reverse()
        return rows


chat_engine = ChatEngine()
