"""Binary up/down/flat forecast strategy backed by the LLM."""
from __future__ import annotations

from typing import Any

from borg.config import settings
from borg.schemas import Direction, ForecastResult
from borg.strategies.base import Strategy, Trade


class BinaryForecastStrategy(Strategy):
    """Forecast direction and execute a binary-style signal if confidence is high enough."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        summary = self._format_summary(candles)

        forecast = await self.llm.analyze_market(symbol, summary)
        confidence = float(forecast.get("confidence", 0.0))
        direction = forecast.get("direction", "flat")
        market_data["last_forecast"] = ForecastResult(
            symbol=symbol,
            direction=Direction(direction),
            confidence=confidence,
            rationale=forecast.get("analysis"),
            model_used=forecast.get("model_used", "unknown"),
            raw_analysis=forecast.get("raw"),
        )

        threshold = self.config.get("confidence_threshold", settings.confidence_threshold)
        if direction == "up" and confidence >= threshold:
            return [Trade(symbol=symbol, side="buy", quantity=1.0, strategy_id=self.name, confidence=confidence)]
        if direction == "down" and confidence >= threshold:
            return [Trade(symbol=symbol, side="sell", quantity=1.0, strategy_id=self.name, confidence=confidence)]
        return []

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 5.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
