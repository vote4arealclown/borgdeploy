"""Dollar-cost averaging strategy: accumulate at fixed intervals.

Generates a small buy signal once per configured interval regardless of price,
disciplining volatility impact through systematic investment.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from borg.strategies.base import Strategy, Trade


class DCAStrategy(Strategy):
    """Generate a periodic buy signal for dollar-cost averaging."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        interval_minutes = self.config.get("interval_minutes", 60)

        if not candles:
            return []

        latest_ts = candles[0].get("ts")
        if isinstance(latest_ts, str):
            latest_ts = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
        if not isinstance(latest_ts, datetime):
            return []

        if latest_ts.minute % interval_minutes != 0:
            return []

        current = float(candles[0]["close"])
        return [
            Trade(
                symbol=symbol,
                side="buy",
                quantity=self.config.get("quantity", 0.25),
                entry_price=current,
                strategy_id=self.name,
                confidence=55.0,
            )
        ]

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 2.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
