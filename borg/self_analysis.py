"""Self-analysis engine: analyze own performance and surface insights."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from borg.db import Database
from borg.strategies.base import Strategy


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = _mean(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) or 1e-9
    return mean / std


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return 0.0
    peak = 0.0
    drawdown = 0.0
    cumulative = 0.0
    for r in returns:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > drawdown:
            drawdown = dd
    return drawdown


class SelfAnalysisEngine:
    """Analyze the agent's own performance from episodes."""

    def __init__(self, database: Database, strategies: list[Strategy] | None = None) -> None:
        self.db = database
        self.strategies = strategies or []

    def analyze_performance(self, period: str = "daily") -> dict[str, Any]:
        """Aggregate performance over daily/weekly/monthly window."""
        if period == "daily":
            delta = timedelta(days=1)
        elif period == "weekly":
            delta = timedelta(days=7)
        else:
            delta = timedelta(days=30)

        since = (datetime.now(timezone.utc) - delta).isoformat()
        episodes = self.db.query_episodes(since=since, limit=10_000)

        if not episodes:
            return {"period": period, "trades_count": 0, "note": "No trades in period"}

        wins = sum(1 for e in episodes if e.get("outcome", {}).get("win"))
        losses = len(episodes) - wins
        returns = [
            float(e["outcome"]["horizon_return"])
            for e in episodes
            if e.get("outcome", {}).get("horizon_return") is not None
        ]
        win_returns = [r for r, e in zip(returns, episodes) if e.get("outcome", {}).get("win")]
        loss_returns = [r for r, e in zip(returns, episodes) if not e.get("outcome", {}).get("win")]

        analysis: dict[str, Any] = {
            "period": period,
            "trades_count": len(episodes),
            "win_count": wins,
            "loss_count": losses,
            "win_rate": wins / len(episodes),
            "avg_win": _mean(win_returns),
            "avg_loss": _mean(loss_returns),
            "total_pnl": _mean(returns) * len(returns),
            "sharpe": _sharpe(returns),
            "max_dd": _max_drawdown(returns),
        }

        by_strategy: dict[str, Any] = {}
        strategy_names = {s.name for s in self.strategies}
        for e in episodes:
            actor = e.get("actor", "unknown")
            strategy_names.add(actor)
        for name in strategy_names:
            strat_trades = [e for e in episodes if e.get("actor") == name]
            if strat_trades:
                sw = sum(1 for e in strat_trades if e.get("outcome", {}).get("win"))
                sr = [
                    float(e["outcome"]["horizon_return"])
                    for e in strat_trades
                    if e.get("outcome", {}).get("horizon_return") is not None
                ]
                by_strategy[name] = {
                    "count": len(strat_trades),
                    "win_rate": sw / len(strat_trades),
                    "total_pnl": _mean(sr) * len(sr),
                }

        by_regime: dict[str, Any] = {}
        for regime in set(e.get("regime", "unknown") for e in episodes):
            regime_trades = [e for e in episodes if e.get("regime") == regime]
            rw = sum(1 for e in regime_trades if e.get("outcome", {}).get("win"))
            rr = [
                float(e["outcome"]["horizon_return"])
                for e in regime_trades
                if e.get("outcome", {}).get("horizon_return") is not None
            ]
            by_regime[regime] = {
                "count": len(regime_trades),
                "win_rate": rw / len(regime_trades),
                "total_pnl": _mean(rr) * len(rr),
            }

        analysis["by_strategy"] = by_strategy
        analysis["by_regime"] = by_regime
        return analysis

    def identify_insights(self, analysis: dict[str, Any]) -> list[str]:
        """Identify interesting patterns in performance."""
        insights: list[str] = []

        if analysis.get("by_strategy"):
            best = max(analysis["by_strategy"].items(), key=lambda x: x[1]["win_rate"])
            insights.append(
                f"Top performer: {best[0]} with {best[1]['win_rate'] * 100:.0f}% win rate "
                f"({best[1]['count']} trades)"
            )
            worst = min(analysis["by_strategy"].items(), key=lambda x: x[1]["win_rate"])
            insights.append(
                f"Weakest performer: {worst[0]} with {worst[1]['win_rate'] * 100:.0f}% win rate "
                f"({worst[1]['count']} trades)"
            )

        if analysis.get("by_regime"):
            best_regime = max(analysis["by_regime"].items(), key=lambda x: x[1]["win_rate"])
            insights.append(
                f"Best regime: {best_regime[0]} with {best_regime[1]['win_rate'] * 100:.0f}% win rate"
            )
            worst_regime = min(analysis["by_regime"].items(), key=lambda x: x[1]["win_rate"])
            insights.append(
                f"Challenging regime: {worst_regime[0]} with {worst_regime[1]['win_rate'] * 100:.0f}% win rate"
            )

        wr = analysis.get("win_rate", 0.0)
        if wr > 0.6:
            insights.append("Strong performance above 60% win rate")
        elif wr < 0.4:
            insights.append("Below baseline performance; review strategy selection")

        return insights

    def surface_risks(self, analysis: dict[str, Any]) -> list[str]:
        """Identify risks and uncertainties."""
        risks: list[str] = []
        if analysis.get("max_dd", 0) > 0.05:
            risks.append(f"Drawdown {analysis['max_dd'] * 100:.1f}%; above typical")
        if analysis.get("win_rate", 0) < 0.45:
            risks.append("Win rate below 45%; elevated stop-hit risk")
        if analysis.get("sharpe", 0) < 1.0:
            risks.append("Sharpe < 1.0; risk-adjusted returns below 1.0 per unit volatility")
        if analysis.get("trades_count", 0) < 5:
            risks.append("Low sample size; statistics may be unreliable")
        return risks
