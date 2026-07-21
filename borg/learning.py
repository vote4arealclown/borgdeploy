"""Learning engine: propose strategy weight updates from backtest results."""
from __future__ import annotations

import logging
from typing import Any

from borg.coordinator import StrategyCoordinator

logger = logging.getLogger(__name__)


def _win_rate(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.get("win")) / len(trades)


def _sharpe_from_trades(trades: list[dict[str, Any]]) -> float:
    import math

    returns = [t["horizon_return"] for t in trades if t.get("horizon_return") is not None]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) or 1e-9
    return mean / std


class LearningEngine:
    """Analyze backtests and propose conservative strategy weight updates."""

    def __init__(self, coordinator: StrategyCoordinator) -> None:
        self.coordinator = coordinator

    def compute_regime_gradients(self, backtest_results: dict[str, Any]) -> dict[str, Any]:
        """Compute performance gradient for each (strategy, regime) pair."""
        gradients: dict[str, Any] = {}
        trades = backtest_results.get("trades", [])

        for trade in trades:
            regime = trade.get("regime", "unknown")
            reasoning = trade.get("reasoning", {})
            for signal in reasoning.get("signals_input", []):
                strategy = signal.get("strategy", "unknown")
                key = f"{strategy}::{regime}"
                if key not in gradients:
                    gradients[key] = {"strategy": strategy, "regime": regime, "trades": []}
                gradients[key]["trades"].append(trade)

        result: dict[str, Any] = {}
        for key, data in gradients.items():
            strat_trades = data["trades"]
            if len(strat_trades) < 3:
                continue
            win_rate = _win_rate(strat_trades)
            sharpe = _sharpe_from_trades(strat_trades)
            gradient = (win_rate - 0.5) + (sharpe - 1.0) * 0.5
            result[key] = {
                "strategy": data["strategy"],
                "regime": data["regime"],
                "win_rate": win_rate,
                "sharpe": sharpe,
                "gradient": gradient,
                "sample_size": len(strat_trades),
            }

        return result

    def propose_weight_updates(
        self,
        backtest_results: dict[str, Any],
        current_weights: dict[tuple[str, str], float],
    ) -> dict[tuple[str, str], float]:
        """Propose new weights by moving 30% toward the gradient direction."""
        gradients = self.compute_regime_gradients(backtest_results)
        proposed: dict[tuple[str, str], float] = {}

        for key, data in gradients.items():
            strategy = data["strategy"]
            regime = data["regime"]
            weight_key = (strategy, regime)
            current_weight = current_weights.get(weight_key, 1.0)
            gradient = data["gradient"]
            delta = gradient * 0.3
            new_weight = current_weight + delta
            new_weight = max(0.5, min(2.0, new_weight))
            proposed[weight_key] = round(new_weight, 4)

        return proposed

    def validate_updates(
        self,
        updates: dict[tuple[str, str], float],
        train_results: dict[str, Any],
        validation_results: dict[str, Any],
    ) -> bool:
        """Reject updates if in-sample Sharpe is much better than out-of-sample."""
        is_sharpe = train_results.get("sharpe", 0.0)
        oos_sharpe = validation_results.get("sharpe", 0.0)
        if oos_sharpe <= 0:
            # If validation is negative or zero, require IS not dramatically positive.
            if is_sharpe > 0.5:
                logger.warning(
                    "Validation Sharpe non-positive (%.3f) while training Sharpe positive (%.3f); likely overfit",
                    oos_sharpe,
                    is_sharpe,
                )
                return False
        if is_sharpe > 1.5 * max(oos_sharpe, 1e-9):
            logger.warning(
                "Overfitting detected: IS Sharpe %.3f vs OOS %.3f",
                is_sharpe,
                oos_sharpe,
            )
            return False
        return True
