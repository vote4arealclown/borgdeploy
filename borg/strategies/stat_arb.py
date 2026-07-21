"""Statistical arbitrage strategy: trade mean-reversion of a z-score."""
from __future__ import annotations

import math
from typing import Any

from borg.strategies.base import Strategy, Trade


class StatArbStrategy(Strategy):
    """Generate trades when price deviates significantly from its recent mean."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        lookback = self.config.get("lookback_period", 30)
        z_threshold = self.config.get("z_threshold", 1.5)

        if len(candles) < lookback:
            return []

        window = candles[:lookback]
        closes = [float(c["close"]) for c in window]
        mean = sum(closes) / lookback
        variance = sum((c - mean) ** 2 for c in closes) / lookback
        std = math.sqrt(variance)
        current = closes[0]
        z_score = (current - mean) / std if std else 0.0

        trades: list[Trade] = []
        if z_score > z_threshold:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(z_score) * 10.0, 95.0),
                )
            )
        elif z_score < -z_threshold:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(z_score) * 10.0, 95.0),
                )
            )
        return trades

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 4.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
