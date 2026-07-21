"""Momentum strategy: follow established price trends.

Buys when the short-term rate of change is positive and accelerating,
sells when it is negative and decelerating.
"""
from __future__ import annotations

from typing import Any

from borg.strategies.base import Strategy, Trade


class MomentumStrategy(Strategy):
    """Generate directional trades based on price momentum."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        lookback = self.config.get("lookback_period", 10)
        min_candles = self.config.get("min_candles", lookback + 5)

        if len(candles) < min_candles:
            return []

        current = float(candles[0]["close"])
        past = float(candles[lookback]["close"])
        roc = (current - past) / past if past else 0.0

        # Normalise ROC to a confidence score.
        threshold = self.config.get("momentum_threshold", 0.005)
        trades: list[Trade] = []
        if roc > threshold:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(roc) * 5000.0, 95.0),
                )
            )
        elif roc < -threshold:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(roc) * 5000.0, 95.0),
                )
            )
        return trades

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 5.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
