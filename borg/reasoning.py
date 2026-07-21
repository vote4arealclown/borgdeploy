"""LLM reasoning layer for trade/forecast decisions."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from borg.llm import llm
from borg.memory import Memory
from borg.schemas import Episode

logger = logging.getLogger(__name__)


class ReasoningPrompt:
    """Template for reasoning about strategy signals."""

    TEMPLATE = """You are an autonomous market-analysis agent reasoning about a potential forecast.

MARKET STATE:
- Asset: {asset}
- Current Price: {current_price}
- Recent OHLCV: High {high}, Low {low}, Close {close}, Volume {volume}
- Features: {features}
- Time: {timestamp}

DETECTED REGIME: {regime}

STRATEGY SIGNALS:
{signals_list}

HISTORICAL EFFICACY (in current regime, last 30 days):
{efficacy_summary}

SIMILAR PAST EPISODES:
{similar_episodes}

YOUR REASONING TASK:
1. Which strategy signal is most reliable in this regime?
2. How much should we size this position (multiplier 0.5-2.0)?
3. What is your confidence (0-100) in this direction?
4. Where should the stop be? (based on regime volatility, as a percent)
5. What's the target? (as a percent)
6. What could go wrong? (risks to watch)

RESPOND ONLY WITH VALID JSON:
{{
    "decision": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0-100.0,
    "size_multiplier": 0.5-2.0,
    "reasoning": "Clear explanation of why (2-3 sentences)",
    "stop_loss_percent": 2.0,
    "take_profit_percent": 5.0,
    "risks": ["risk1", "risk2"],
    "overrides_strategy_vote": true/false
}}
"""

    @staticmethod
    def build(
        market_data: dict[str, Any],
        signals: list[dict[str, Any]],
        regime: str,
        efficacy_data: dict[str, Any],
        similar_episodes: list[Episode],
    ) -> str:
        """Populate the reasoning prompt with current context."""
        candles = market_data.get("candles", [])
        latest = candles[0] if candles else {}
        features = market_data.get("features", {})

        signals_list = "\n".join(
            f"- {s.get('strategy', 'unknown')}: {s.get('action', 'HOLD')} "
            f"(confidence: {s.get('confidence', 0):.1f}, "
            f"efficacy in regime: {efficacy_data.get(s.get('strategy'), {}).get('win_rate', 'N/A')})"
            for s in signals
        ) or "- No signals"

        efficacy_summary = "\n".join(
            f"- {strat}: {data.get('win_rate', 0) * 100:.0f}% wins ({data.get('sample_size', 0)} episodes)"
            for strat, data in efficacy_data.items()
        ) or "- No efficacy data"

        similar_summary = "\n".join(
            f"- {ep.timestamp.isoformat() if ep.timestamp else 'unknown'}: {ep.actor} in {ep.regime}, "
            f"return {ep.outcome.horizon_return if ep.outcome else None}, held to resolution"
            for ep in similar_episodes[:5]
        ) or "- No similar episodes"

        from datetime import datetime

        return ReasoningPrompt.TEMPLATE.format(
            asset=market_data.get("symbol", "UNKNOWN"),
            current_price=latest.get("close", "N/A"),
            high=latest.get("high", "N/A"),
            low=latest.get("low", "N/A"),
            close=latest.get("close", "N/A"),
            volume=latest.get("volume", "N/A"),
            features=json.dumps(features, default=str),
            timestamp=datetime.now().isoformat(),
            regime=regime,
            signals_list=signals_list,
            efficacy_summary=efficacy_summary,
            similar_episodes=similar_summary,
        )


class ReasoningEngine:
    """Reason about strategy signals and return structured decisions."""

    def __init__(self, llm_client: Any = llm, memory: Optional[Memory] = None) -> None:
        self.llm = llm_client
        self.memory = memory

    def _fallback_reasoning(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Deterministic fallback when the LLM is unavailable or fails."""
        action = signal.get("action", "HOLD")
        return {
            "decision": action,
            "confidence": signal.get("confidence", 50.0),
            "size_multiplier": 1.0,
            "reasoning": "LLM reasoning unavailable; using strategy signal directly",
            "stop_loss_percent": 2.0,
            "take_profit_percent": 5.0,
            "risks": ["LLM unavailable; deterministic fallback active"],
            "overrides_strategy_vote": False,
        }

    def _parse_response(self, raw: str, default_signal: dict[str, Any]) -> dict[str, Any]:
        """Extract JSON from an LLM response."""
        try:
            match = raw.strip()
            # Handle possible markdown fences.
            if "```" in match:
                match = match.split("```")[1]
                if match.startswith("json"):
                    match = match[3:]
            parsed = json.loads(match)
        except Exception as exc:
            logger.warning("Failed to parse reasoning JSON: %s", exc)
            return self._fallback_reasoning(default_signal)

        # Normalize decision to BUY/SELL/HOLD.
        decision = str(parsed.get("decision", "HOLD")).upper()
        if decision not in {"BUY", "SELL", "HOLD"}:
            # Accept up/down/flat mapping.
            decision = {"UP": "BUY", "DOWN": "SELL", "FLAT": "HOLD"}.get(decision, "HOLD")

        return {
            "decision": decision,
            "confidence": max(0.0, min(100.0, float(parsed.get("confidence", default_signal.get("confidence", 50.0))))),
            "size_multiplier": max(0.5, min(2.0, float(parsed.get("size_multiplier", 1.0)))),
            "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
            "stop_loss_percent": float(parsed.get("stop_loss_percent", 2.0)),
            "take_profit_percent": float(parsed.get("take_profit_percent", 5.0)),
            "risks": list(parsed.get("risks", [])) or ["No risks listed"],
            "overrides_strategy_vote": bool(parsed.get("overrides_strategy_vote", False)),
        }

    async def reason(
        self,
        market_data: dict[str, Any],
        signals: list[dict[str, Any]],
        regime: str,
    ) -> dict[str, Any]:
        """Main entry point: reason about signals and return structured output."""
        default_signal = signals[0] if signals else {"action": "HOLD", "confidence": 50.0}

        efficacy_data: dict[str, Any] = {}
        if self.memory:
            for signal in signals:
                strategy = signal.get("strategy", "unknown")
                efficacy = await self.memory.get_strategy_efficacy(strategy, regime, window_days=30)
                efficacy_data[strategy] = efficacy.model_dump()

        similar_episodes: list[Episode] = []
        if self.memory and signals:
            trigger = signals[0].get("trigger", "")
            similar_episodes = await self.memory.find_similar_episodes(regime, trigger, limit=5)

        prompt = ReasoningPrompt.build(
            market_data=market_data,
            signals=signals,
            regime=regime,
            efficacy_data=efficacy_data,
            similar_episodes=similar_episodes,
        )

        try:
            raw = await self.llm.generate(prompt)
            if not raw or raw.startswith("Ollama error"):
                return self._fallback_reasoning(default_signal)
            return self._parse_response(raw, default_signal)
        except Exception as exc:
            logger.warning("Reasoning LLM failed: %s", exc)
            return self._fallback_reasoning(default_signal)
