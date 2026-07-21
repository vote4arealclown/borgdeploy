"""Grid trading strategy: trade within a fixed price grid.

Places buy signals near grid supports and sell signals near grid resistances
around a central reference price.
"""
from __future__ import annotations

from typing import Any

from borg.strategies.base import Strategy, Trade


class GridTradingStrategy(Strategy):
    """Generate trades at fixed intervals above and below a central price."""

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        candles = market_data.get("candles", [])
        lookback = self.config.get("lookback_period", 20)
        grid_levels = self.config.get("grid_levels", 5)
        grid_spacing = self.config.get("grid_spacing", 0.01)

        if len(candles) < lookback:
            return []

        window = candles[:lookback]
        central = sum(float(c["close"]) for c in window) / lookback
        current = float(candles[0]["close"])
        spacing = central * grid_spacing

        trades: list[Trade] = []
        for level in range(1, grid_levels + 1):
            support = central - level * spacing
            resistance = central + level * spacing
            if current <= support * 1.001:
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="buy",
                        quantity=self.config.get("quantity", 1.0),
                        entry_price=current,
                        strategy_id=self.name,
                        confidence=min(50.0 + level * 5.0, 95.0),
                    )
                )
                break
            if current >= resistance * 0.999:
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="sell",
                        quantity=self.config.get("quantity", 1.0),
                        entry_price=current,
                        strategy_id=self.name,
                        confidence=min(50.0 + level * 5.0, 95.0),
                    )
                )
                break
        return trades

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": self.config.get("max_drawdown_pct", 3.0),
            "max_position_size_usd": self.config.get("max_position_size_usd", 100.0),
            "max_leverage": self.config.get("max_leverage", 1.0),
        }
