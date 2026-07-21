"""VWAP strategy: trade price deviations from the volume-weighted average price."""
from __future__ import annotations

from typing import Any

from borg.strategies.base import Strategy, Trade


class VWAPStrategy(Strategy):
    """Buy when price falls below VWAP, sell when it rises above VWAP."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        lookback = self.config.get("lookback_period", 20)
        deviation = self.config.get("deviation_threshold", 0.005)

        if len(candles) < lookback:
            return []

        window = candles[:lookback]
        typical = [(float(c["high"]) + float(c["low"]) + float(c["close"])) / 3.0 for c in window]
        volumes = [float(c["volume"]) for c in window]
        total_value = sum(t * v for t, v in zip(typical, volumes))
        total_volume = sum(volumes)
        vwap = total_value / total_volume if total_volume else 0.0

        current = float(candles[0]["close"])
        pct_dev = (current - vwap) / vwap if vwap else 0.0

        trades: list[Trade] = []
        if pct_dev > deviation:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(pct_dev) * 3000.0, 95.0),
                )
            )
        elif pct_dev < -deviation:
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=self.config.get("quantity", 1.0),
                    entry_price=current,
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(pct_dev) * 3000.0, 95.0),
                )
            )
        return trades

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 3.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
