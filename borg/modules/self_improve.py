"""Self-improvement module: generate proposed code diffs from reflection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from borg.config import settings
from borg.db import Database, db
from borg.events import EventLog, event_log
from borg.llm import llm
from borg.versioning import Versioning, versioning


class SelfImproveModule:
    """Analyzes recent learnings and proposes small code/strategy changes."""

    def __init__(
        self,
        database: Database = db,
        events: EventLog = event_log,
        versioning: Versioning = versioning,
    ) -> None:
        self.db = database
        self.events = events
        self.versioning = versioning
        self.project_root = settings.sqlite_path.parent.parent if settings.sqlite_path else Path(__file__).resolve().parent.parent.parent

    async def analyze(self) -> Optional[dict[str, Any]]:
        """Review recent learnings and decide whether a change is warranted."""
        learnings = self.db.recent_learnings(limit=10)
        if len(learnings) < 3:
            return None

        prompts = [
            "Improve forecast confidence threshold handling",
            "Add a new technical indicator to BinaryForecastStrategy",
            "Make the colony visualization rooms more informative",
            "Improve event log filtering in the dashboard",
        ]

        # Use LLM if available to pick the most promising improvement
        if await llm._check_ollama():
            context = "\n".join(f"- {learning['summary'][:200]}" for learning in learnings)
            options_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
            prompt = f"""Given these recent learnings, pick the single best improvement from the list and explain in one sentence.

Learnings:
{context}

Options:
{options_text}

Choice and reason:"""
            try:
                choice = await llm.generate(prompt, timeout=30.0)
                # Map back to option by index
                for i, p in enumerate(prompts):
                    if str(i + 1) in choice or p.lower() in choice.lower():
                        return await self.prove(p)
            except Exception:
                pass

        # Fallback: deterministic rotation based on learning count
        choice = prompts[len(learnings) % len(prompts)]
        return await self.prove(choice)

    def _already_proposed(self, relative_path: str) -> bool:
        """Return True if an open proposal already exists for this path."""
        for version in self.versioning.list(status="proposed"):
            if version.get("path") == relative_path and version.get("module") == "self_improve":
                return True
        return False

    async def prove(self, improvement: str) -> Optional[dict[str, Any]]:
        """Generate a code diff proposal for the chosen improvement."""
        path_map = {
            "Improve forecast confidence threshold handling": "borg/strategies/binary_forecast.py",
            "Add a new technical indicator to BinaryForecastStrategy": "borg/strategies/binary_forecast.py",
            "Make the colony visualization rooms more informative": "borg/visual/sim.py",
            "Improve event log filtering in the dashboard": "borg/web/templates/dashboard.html",
        }
        relative_path = path_map.get(improvement, "borg/brain.py")

        if self._already_proposed(relative_path):
            return None

        original = (self.project_root / relative_path).read_text(encoding="utf-8")

        if "confidence" in improvement.lower():
            new_content = self._patch_confidence(original)
        elif "indicator" in improvement.lower():
            new_content = self._patch_indicator(original)
        elif "colony" in improvement.lower():
            new_content = self._patch_colony(original)
        elif "dashboard" in improvement.lower():
            new_content = self._patch_dashboard(original)
        else:
            new_content = original + f"\n# TODO: implement '{improvement}'\n"

        if new_content == original:
            return None

        self.events.emit(
            f"Self-improvement proposed: {improvement}",
            category="self_improve",
            phase="plan",
            metadata={"path": relative_path},
        )
        return self.versioning.propose("self_improve", relative_path, new_content)

    def _patch_confidence(self, original: str) -> str:
        marker = "threshold = self.config.get(\"confidence_threshold\", settings.confidence_threshold)"
        if marker in original:
            return original.replace(
                marker,
                "threshold = self.config.get(\"confidence_threshold\", settings.confidence_threshold)\n        # Adaptive: lower threshold in high-volatility regime (self-improvement v1)\n        if len(candles) > 5:\n            recent_range = max(c[\"high\"] for c in candles[:5]) - min(c[\"low\"] for c in candles[:5])\n            avg_close = sum(c[\"close\"] for c in candles[:5]) / 5\n            if recent_range / avg_close > 0.001:\n                threshold = max(50.0, threshold - 5.0)",
            )
        return original

    def _patch_indicator(self, original: str) -> str:
        marker = "async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:"
        if marker in original and "momentum" not in original.lower():
            return original.replace(
                marker,
                "def _momentum(self, candles: list[dict[str, Any]]) -> float:\n        if len(candles) < 3:\n            return 0.0\n        return candles[0][\"close\"] - candles[2][\"close\"]\n\n    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:",
            )
        return original

    def _patch_colony(self, original: str) -> str:
        marker = 'Room("tower", "Monitor Tower", 45, 70, 12, 10, "#ef4444", "📊"),'
        if marker in original and "Hive" not in original:
            return original.replace(
                marker,
                'Room("tower", "Monitor Tower", 45, 70, 12, 10, "#ef4444", "📊"),\n        Room("hive", "Learning Hive", 82, 70, 12, 10, "#ec4899", "🍯"),',
            )
        return original

    def _patch_dashboard(self, original: str) -> str:
        marker = "<h2>Live Event Log</h2>"
        if marker in original and "categoryFilter" not in original:
            return original.replace(
                marker,
                '<h2>Live Event Log</h2>\n            <div id="categoryFilter" style="margin-bottom:0.5rem;">\n                <button onclick="filterEvents(\'all\')">All</button>\n                <button onclick="filterEvents(\'brain\')">Brain</button>\n                <button onclick="filterEvents(\'chat\')">Chat</button>\n                <button onclick="filterEvents(\'system\')">System</button>\n            </div>',
            )
        return original


self_improve = SelfImproveModule()
