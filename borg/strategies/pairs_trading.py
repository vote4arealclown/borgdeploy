"""Pairs trading strategy: simplified spread-reversion signal.

Because Borg receives one symbol at a time, this implementation constructs a
synthetic benchmark from the symbol's own moving average and trades the spread
between current price and that benchmark. A real multi-asset implementation
would compare two correlated symbols loaded together.
"""
from __future__ import annotations

from typing import Any

from borg.strategies.base import Strategy, Trade


class PairsTradingStrategy(Strategy):
    """Generate trades when price diverges from a synthetic benchmark spread."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        lookback = self.config.get("lookback_period", 30)
        threshold = self.config.get("spread_threshold", 0.02)

        if len(candles) < lookback + 1:
            return []

        window = candles[:lookback]
        benchmark = sum(float(c["close"]) for c in window) / lookback
        current = float(candles[0]["close"])
        prior = float(candles[1]["close"])

        # Synthetic spread: current price vs benchmark.
        spread = (current - benchmark) / benchmark if benchmark else 0.0
        # Momentum of the spread.
        spread_change = (current - prior) / prior if prior else 0.0

        trades: list[Trade] = []
        if spread > threshold and spread_change < 0:
            # Spread is wide and starting to revert; buy underperforming leg.
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(spread) * 1500.0, 95.0),
                )
            )
        elif spread < -threshold and spread_change > 0:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(spread) * 1500.0, 95.0),
                )
            )
        return trades

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 4.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
