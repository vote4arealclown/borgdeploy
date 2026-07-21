"""Consensus meta-strategy: require agreement among sub-strategies."""
from __future__ import annotations

from typing import Any

from borg.strategies.base import Strategy, Trade


class ConsensusStrategy(Strategy):
    """Aggregate signals from sub-strategies; only trade when enough agree."""

    def __init__(self, name: str, config: dict[str, Any], strategies: list[Strategy]) -> None:
        super().__init__(name, config)
        self.strategies = strategies

    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        min_votes = self.config.get("min_votes", 2)
        all_trades: list[Trade] = []

        for strategy in self.strategies:
            if not strategy.config.get("enabled", True):
                continue
            strategy.db = self.db
            strategy.llm = self.llm
            try:
                trades = await strategy.analyze(market_data)
                all_trades.extend(trades)
            except Exception as exc:
                # Log and continue; one failed strategy should not kill consensus.
                if self.db is not None:
                    from borg.events import event_log

                    event_log.emit(
                        f"Consensus sub-strategy {strategy.name} failed: {exc}",
                        category="strategy",
                        phase="plan",
                        symbol=symbol,
                    )

        buy_votes = [t for t in all_trades if t.side == "buy"]
        sell_votes = [t for t in all_trades if t.side == "sell"]

        trades: list[Trade] = []
        if len(buy_votes) >= min_votes:
            avg_conf = sum(t.confidence for t in buy_votes) / len(buy_votes)
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=sum(t.quantity for t in buy_votes) / len(buy_votes),
                    entry_price=buy_votes[0].entry_price,
                    strategy_id=f"{self.name} ({len(buy_votes)} votes)",
                    confidence=avg_conf,
                )
            )
        elif len(sell_votes) >= min_votes:
            avg_conf = sum(t.confidence for t in sell_votes) / len(sell_votes)
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=sum(t.quantity for t in sell_votes) / len(sell_votes),
                    entry_price=sell_votes[0].entry_price,
                    strategy_id=f"{self.name} ({len(sell_votes)} votes)",
                    confidence=avg_conf,
                )
            )

        return trades

    def risk_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "max_drawdown_pct": float("inf"),
            "max_position_size_usd": float("inf"),
            "max_leverage": float("inf"),
        }
        for strategy in self.strategies:
            sub = strategy.risk_metrics()
            for key in metrics:
                metrics[key] = min(metrics[key], sub.get(key, float("inf")))
        return metrics
