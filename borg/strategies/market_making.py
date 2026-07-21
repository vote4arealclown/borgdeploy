"""Market-making strategy: capture oscillations around the mid-price.

Without live order-book data, this implementation treats the recent price range
as a synthetic bid-ask channel and buys near the bottom of the range and sells
near the top.
"""
from __future__ import annotations

from typing import Any

from borg.strategies.base import Strategy, Trade


class MarketMakingStrategy(Strategy):
    """Generate buy/sell signals near the edges of a recent price range."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        lookback = self.config.get("lookback_period", 20)
        threshold = self.config.get("range_threshold", 0.2)

        if len(candles) < lookback:
            return []

        window = candles[:lookback]
        highs = [float(c["high"]) for c in window]
        lows = [float(c["low"]) for c in window]
        highest = max(highs)
        lowest = min(lows)
        mid = (highest + lowest) / 2.0
        current = float(candles[0]["close"])
        range_size = highest - lowest
        if range_size == 0:
            return []

        position = (current - lowest) / range_size
        trades: list[Trade] = []
        if position >= 1.0 - threshold:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + (position - 0.5) * 80.0, 95.0),
                )
            )
        elif position <= threshold:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + (0.5 - position) * 80.0, 95.0),
                )
            )
        return trades

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 2.5),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
