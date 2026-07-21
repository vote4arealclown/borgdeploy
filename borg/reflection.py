"""Reflection engine: query past episodes to inform current decisions."""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from borg.llm import llm
from borg.memory import Memory
from borg.schemas import StrategyEfficacy

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Reflect on similar past episodes and strategy efficacy in a regime."""

    def __init__(self, memory: Memory, llm_client: Any = llm) -> None:
        self.memory = memory
        self.llm = llm_client

    async def reflect_on_signal(
        self,
        current_signal: dict[str, Any],
        regime: str,
    ) -> dict[str, Any]:
        """Return confidence adjustment and reasoning from historical episodes."""
        strategy = current_signal.get("strategy", "unknown")
        trigger = current_signal.get("trigger", "")
        base_confidence = float(current_signal.get("confidence", 50.0))

        similar = await self.memory.find_similar_episodes(
            current_regime=regime,
            trigger=trigger,
            limit=10,
        )

        if not similar:
            return {
                "reflection": "No historical precedent in memory",
                "confidence_adj": 1.0,
                "win_rate_in_regime": None,
                "sample_size": 0,
                "warnings": [],
            }

        wins = sum(1 for ep in similar if ep.outcome and ep.outcome.win)
        win_rate = wins / len(similar)
        bars_held = [
            ep.outcome.horizon_return
            for ep in similar
            if ep.outcome and ep.outcome.horizon_return is not None
        ]
        losses = [
            abs(ep.outcome.horizon_return or 0.0)
            for ep in similar
            if ep.outcome and (ep.outcome.horizon_return or 0.0) < 0
        ]
        max_dd = max(losses) if losses else 0.0
        avg_hold = sum(bars_held) / len(bars_held) if bars_held else 0.0

        warnings: list[str] = []
        reflection_text = (
            f"Strategy has {win_rate * 100:.0f}% win rate in this regime "
            f"across {len(similar)} similar episodes."
        )

        if win_rate < 0.5:
            warnings.append(f"Historical win rate below 50% ({win_rate * 100:.0f}%)")
            try:
                reflection_text = await self.llm.generate(
                    f"""Strategy: {strategy}
Regime: {regime}
Past {len(similar)} similar episodes:
- Win rate: {win_rate * 100:.0f}%
- Avg horizon return: {avg_hold:.4f}
- Max adverse return: {max_dd:.4f}
Current confidence: {base_confidence:.1f}%

Briefly: should we proceed, and what risks should we watch?""",
                )
            except Exception as exc:
                logger.warning("LLM reflection failed: %s", exc)
                reflection_text = f"Low historical win rate ({win_rate * 100:.0f}%)."

        if max_dd > 0.02:
            warnings.append(f"Similar episodes saw max adverse return {max_dd * 100:.1f}%")

        # Boost confidence if historical win rate is high, dampen if low.
        confidence_adj = min(1.2, max(0.7, 1.0 + (win_rate - 0.5)))

        return {
            "reflection": reflection_text,
            "win_rate_in_regime": win_rate,
            "confidence_adj": confidence_adj,
            "sample_size": len(similar),
            "avg_horizon_return": avg_hold,
            "max_adverse_return": max_dd,
            "warnings": warnings,
        }

    async def periodic_reflection(
        self,
        window_hours: int = 24,
    ) -> dict[str, Any]:
        """Summarize recent episodes grouped by strategy and regime."""
        since = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
        episodes = self.memory.db.query_episodes(since=since, limit=10_000)

        if not episodes:
            return {"note": "No episodes in the reflection window", "reflections": {}}

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for ep in episodes:
            grouped[(ep["actor"], ep["regime"])].append(ep)

        reflections: dict[str, Any] = {}
        for (strategy, regime), eps in grouped.items():
            wins = sum(1 for e in eps if e.get("outcome", {}).get("win"))
            returns = [
                float(e["outcome"]["horizon_return"])
                for e in eps
                if e.get("outcome", {}).get("horizon_return") is not None
            ]
            reflections[f"{strategy}::{regime}"] = {
                "strategy": strategy,
                "regime": regime,
                "trades": len(eps),
                "wins": wins,
                "losses": len(eps) - wins,
                "win_rate": wins / len(eps),
                "avg_horizon_return": sum(returns) / len(returns) if returns else 0.0,
            }

        return {"window_hours": window_hours, "reflections": reflections}

    async def efficacy_summary(
        self,
        strategies: list[str],
        regime: str,
        window_days: int = 30,
    ) -> dict[str, StrategyEfficacy]:
        """Fetch efficacy for every strategy in the given regime."""
        summary: dict[str, StrategyEfficacy] = {}
        for strategy in strategies:
            efficacy = await self.memory.get_strategy_efficacy(strategy, regime, window_days)
            summary[strategy] = efficacy
        return summary
