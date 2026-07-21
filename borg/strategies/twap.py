"""TWAP-style strategy: distribute a theoretical order over time.

This implementation produces a small periodic buy signal when the configured
interval has elapsed, emulating the disciplined accumulation behaviour of TWAP.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from borg.strategies.base import Strategy, Trade


class TWAPStrategy(Strategy):
    """Generate a small buy signal at regular time intervals."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        interval_minutes = self.config.get("interval_minutes", 60)

        if not candles:
            return []

        # Use the latest candle timestamp to decide whether this is a TWAP slot.
        latest_ts = candles[0].get("ts")
        if isinstance(latest_ts, str):
            latest_ts = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
        if not isinstance(latest_ts, datetime):
            return []

        minute = latest_ts.minute
        if minute % interval_minutes != 0:
            return []

        current = float(candles[0]["close"])
        return [
            Trade(
                symbol=symbol,
                side="buy",
                quantity=self.config.get("quantity", 0.5),
                entry_price=current,
                strategy_id=self.name,
                confidence=50.0,
            )
        ]

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 2.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
