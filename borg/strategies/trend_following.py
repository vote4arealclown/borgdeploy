"""Trend-following strategy: ride established trends via moving-average crossover."""
from __future__ import annotations

from typing import Any

from borg.strategies.base import Strategy, Trade


class TrendFollowingStrategy(Strategy):
    """Generate trades when a fast moving average crosses a slow moving average."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        fast_period = self.config.get("fast_period", 10)
        slow_period = self.config.get("slow_period", 30)

        if len(candles) < slow_period + 1:
            return []

        def ma(period: int, offset: int = 0) -> float:
            window = candles[offset : offset + period]
            return sum(float(c["close"]) for c in window) / period

        fast_now = ma(fast_period, 0)
        slow_now = ma(slow_period, 0)
        fast_prev = ma(fast_period, 1)
        slow_prev = ma(slow_period, 1)

        current = float(candles[0]["close"])
        trades: list[Trade] = []

        # Bullish crossover: fast crosses above slow.
        if fast_prev <= slow_prev and fast_now > slow_now:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=70.0,
                )
            )
        # Bearish crossover: fast crosses below slow.
        elif fast_prev >= slow_prev and fast_now < slow_now:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=70.0,
                )
            )
        return trades

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 5.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
