"""Mean-reversion strategy: trade deviations from a simple moving average."""
from __future__ import annotations

from typing import Any

from borg.strategies.base import Strategy, Trade


class MeanReversionStrategy(Strategy):
    """Generate trades when price deviates from its moving average."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        ma_period = self.config.get("ma_period", 20)
        deviation = self.config.get("deviation_threshold", 0.03)

        if len(candles) < ma_period:
            return []

        # Candles are returned newest-first; use the most recent `ma_period`.
        window = candles[:ma_period]
        ma = sum(float(c["close"]) for c in window) / ma_period
        current = float(candles[0]["close"])
        pct_dev = (current - ma) / ma if ma else 0.0

        trades: list[Trade] = []
        if pct_dev > deviation:
            # Price above MA; expect reversion down.
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=self.config.get("quantity", 2.0),
                    entry_price=current,
                    stop_loss=ma * (1 + deviation * 1.5),
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(pct_dev) * 1000.0, 95.0),
                )
            )
        elif pct_dev < -deviation:
            # Price below MA; expect reversion up.
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=self.config.get("quantity", 2.0),
                    entry_price=current,
                    stop_loss=ma * (1 - deviation * 1.5),
                    strategy_id=self.name,
                    confidence=min(50.0 + abs(pct_dev) * 1000.0, 95.0),
                )
            )

        return trades

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 3.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 150.0),
            "max_leverage": self.config.get("max_leverage", 1.5),
        }
