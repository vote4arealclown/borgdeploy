"""Backtesting engine: run historical simulations and record episodes."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from borg.coordinator import StrategyCoordinator
from borg.db import Database
from borg.episode import episode_from_forecast
from borg.memory import Memory
from borg.regime import detect_regime
from borg.schemas import Episode, EpisodeOutcome, EpisodeSignal


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
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


class BacktestEngine:
    """Run historical backtests using stored candles and record outcomes."""

    def __init__(
        self,
        coordinator: StrategyCoordinator,
        database: Database,
        memory: Optional[Memory] = None,
    ) -> None:
        self.coordinator = coordinator
        self.db = database
        self.memory = memory

    async def run_backtest(
        self,
        start: datetime,
        end: datetime,
        symbol: Optional[str] = None,
        record_episodes: bool = True,
    ) -> dict[str, Any]:
        """Walk historical candles and simulate forecasts."""
        symbols = [symbol] if symbol else []
        if not symbols:
            # Distinct symbols in range.
            rows = self.db.fetchall(
                "SELECT DISTINCT symbol FROM market_candles WHERE ts >= %s AND ts <= %s"
                if self.db.is_postgres
                else "SELECT DISTINCT symbol FROM market_candles WHERE ts >= ? AND ts <= ?",
                (start.isoformat(), end.isoformat()),
            )
            symbols = [r["symbol"] for r in rows]

        all_trades: list[dict[str, Any]] = []
        symbol_results: dict[str, list[dict[str, Any]]] = {}

        for sym in symbols:
            candles = self.db.fetchall(
                "SELECT * FROM market_candles WHERE symbol = %s AND ts >= %s AND ts <= %s ORDER BY ts"
                if self.db.is_postgres
                else "SELECT * FROM market_candles WHERE symbol = ? AND ts >= ? AND ts <= ? ORDER BY ts",
                (sym, start.isoformat(), end.isoformat()),
            )
            candles = [dict(r) for r in candles]
            trades = await self._simulate_symbol(sym, candles, record_episodes)
            symbol_results[sym] = trades
            all_trades.extend(trades)

        returns = [t["horizon_return"] for t in all_trades if t.get("horizon_return") is not None]
        wins = sum(1 for t in all_trades if t.get("win"))

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": symbols,
            "trades": all_trades,
            "trade_count": len(all_trades),
            "win_rate": wins / len(all_trades) if all_trades else 0.0,
            "sharpe": _sharpe(returns),
            "max_dd": _max_drawdown(returns),
            "total_return": sum(returns) if returns else 0.0,
            "symbol_results": symbol_results,
        }

    async def _simulate_symbol(
        self,
        symbol: str,
        candles: list[dict[str, Any]],
        record_episodes: bool,
        min_candles: int = 20,
    ) -> list[dict[str, Any]]:
        """Simulate forecasts for a single symbol's candles."""
        trades: list[dict[str, Any]] = []
        if len(candles) < min_candles + 1:
            return trades

        for i in range(min_candles, len(candles) - 1):
            window = list(reversed(candles[:i]))  # newest-first
            current = candles[i]
            next_candle = candles[i + 1]
            market_data = {
                "symbol": symbol,
                "candles": window,
                "features": {},
            }
            regime = detect_regime(window)
            reasoning = await self.coordinator.coordinate(market_data, regime=regime)
            decision = reasoning.get("decision", "HOLD")
            if decision == "HOLD":
                continue

            direction = "up" if decision == "BUY" else "down"
            entry_price = float(current["close"])
            exit_price = float(next_candle["close"])
            if direction == "up":
                correct = exit_price > entry_price
            else:
                correct = exit_price < entry_price
            horizon_return = (exit_price - entry_price) / entry_price
            if direction == "down":
                horizon_return = -horizon_return
            win = correct

            trade = {
                "symbol": symbol,
                "timestamp": current["ts"],
                "regime": regime,
                "direction": direction,
                "confidence": reasoning.get("confidence", 0.0),
                "reasoning": reasoning,
                "entry": entry_price,
                "exit": exit_price,
                "horizon_return": horizon_return,
                "win": win,
            }
            trades.append(trade)

            if record_episodes and self.memory:
                forecast = {
                    "created_at": current["ts"],
                    "direction": direction,
                    "confidence": reasoning.get("confidence", 0.0),
                    "rationale": reasoning.get("reasoning"),
                    "model_used": "backtest",
                }
                episode = episode_from_forecast(
                    forecast,
                    market_data,
                    actor="backtest",
                    outcome={
                        "win": win,
                        "correct": correct,
                        "resolved_at": next_candle["ts"],
                        "horizon_return": horizon_return,
                        "outcome": "win" if win else "loss",
                    },
                )
                episode.reasoning_output = reasoning
                try:
                    await self.memory.store_episode(episode)
                except Exception as exc:
                    # Backtest episode storage is best-effort.
                    import logging

                    logging.getLogger(__name__).warning("Backtest episode storage failed: %s", exc)

        return trades
