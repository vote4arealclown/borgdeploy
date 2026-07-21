"""Borg OODA loop: Observe → Orient/Plan → Act → Reflect."""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from borg.config import settings
from borg.coordinator import StrategyCoordinator
from borg.db import Database, db
from borg.distributed import DistributedLock
from borg.backtest_engine import BacktestEngine
from borg.consciousness_reporter import ConsciousnessReporter
from borg.consciousness_score import ConsciousnessScore
from borg.episode import episode_from_forecast
from borg.events import EventLog, event_log
from borg.learning import LearningEngine
from borg.llm import llm
from borg.llm_hybrid import hybrid_llm, should_use_hybrid
from borg.memory import Memory, memory
from borg.metrics import (
    brain_cycle_duration_seconds,
    brain_cycles_total,
    errors_total,
    forecasts_generated,
    llm_inference_latency_seconds,
    record_resource_usage,
    timed,
)
from borg.modules.binary_options import analyzer
from borg.modules.databricks_export import export_all
from borg.modules.hip4 import get_daily_binaries
from borg.modules.reports import report_engine
from borg.modules.hip4_paper_trader import create_daily_paper_trade, settle_open_paper_trades
from borg.modules.market_data import fetch_latest_candles
from borg.modules.self_improve import self_improve
from borg.modules.diary import diary_writer
from borg.monitor import monitor
from borg.reasoning_audit import ReasoningAudit
from borg.schemas import CandleInput, Direction, ForecastResult
from borg.self_analysis import SelfAnalysisEngine
from borg.strategies.base import Strategy, Trade
from borg.strategies.binary_forecast import BinaryForecastStrategy
from borg.visual.sim import ColonySim, colony

logger = logging.getLogger(__name__)


class DataFeed:
    """Generate synthetic candles so the prototype runs without external market data."""

    def __init__(self, database: Database = db) -> None:
        self._db = database
        # Seed prices for synthetic fallback; real data overwrites these via
        # the DB unique (symbol, ts) constraint as it arrives.
        self._prices: dict[str, float] = {
            "BTC": 65_000.0,
            "ETH": 3_400.0,
            "SOL": 150.0,
            "XLM": 0.18,
            "XRP": 1.10,
            "BNB": 570.0,
            "HYPE": 60.0,
            "EURUSD": 1.14,
            "GBPUSD": 1.34,
            "USDJPY": 162.0,
        }

    async def _seed_real_history(self, database: Database) -> bool:
        """Try to seed the last ~30 minutes of real 1m candles."""
        try:
            candles_by_symbol = await fetch_latest_candles(period="1d", interval="1m")
        except Exception as exc:
            logger.warning("Real market data fetch failed during seed: %s", exc)
            return False

        total = 0
        for symbol, candles in candles_by_symbol.items():
            # Keep the newest 30 real candles so the feature window is populated.
            for candle in candles[-30:]:
                database.insert_candle(candle.model_dump())
                total += 1
        if total:
            logger.info("Seeded %s real 1m candles", total)
            return True
        return False

    def seed(self, database: Optional[Database] = None) -> None:
        """Create a small historical window for each configured symbol."""
        db = database or self._db
        # Real-data seeding is async; the caller (brain_loop) awaits it via refresh below.
        # Fall back to synthetic history if real data is not available yet.
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        for symbol in settings.symbol_list:
            price = self._prices.get(symbol, 100.0)
            for i in range(30, 0, -1):
                ts = now - timedelta(minutes=i)
                open_p = price * (1 + random.gauss(0, 0.001))
                close_p = open_p * (1 + random.gauss(0, 0.0008))
                high_p = max(open_p, close_p) * (1 + abs(random.gauss(0, 0.0005)))
                low_p = min(open_p, close_p) * (1 - abs(random.gauss(0, 0.0005)))
                db.insert_candle(
                    CandleInput(
                        symbol=symbol,
                        ts=ts,
                        open=open_p,
                        high=high_p,
                        low=low_p,
                        close=close_p,
                        volume=abs(random.gauss(1_000_000, 200_000)),
                    ).model_dump()
                )

    async def refresh_real_data(self, database: Optional[Database] = None) -> int:
        """Fetch and insert the latest real 1m candles for all symbols.

        Returns the number of new candles inserted.  Duplicates are ignored by
        the database's unique (symbol, ts) constraint.
        """
        db = database or self._db
        # Use a 4-hour lookback so low-volume builder markets (xyz forex)
        # still get a usable window when 30m of data is sparse.
        try:
            candles_by_symbol = await fetch_latest_candles(period="4h", interval="1m")
        except Exception as exc:
            logger.warning("Real market data refresh failed: %s", exc)
            return 0

        total = 0
        for symbol, candles in candles_by_symbol.items():
            for candle in candles:
                try:
                    db.insert_candle(candle.model_dump())
                    total += 1
                except Exception:
                    # Duplicate or DB error; ignore and continue.
                    pass
        if total:
            logger.info("Refreshed %s real 1m candles", total)
        return total

    def next_candle(self, symbol: str) -> dict[str, Any]:
        """Produce one new synthetic candle."""
        prev = self._db.latest_candles(symbol, limit=1)
        prev_price = float(prev[0]["close"]) if prev else self._prices.get(symbol, 100.0)
        open_p = prev_price * (1 + random.gauss(0, 0.0005))
        close_p = open_p * (1 + random.gauss(0, 0.001))
        high_p = max(open_p, close_p) * (1 + abs(random.gauss(0, 0.0006)))
        low_p = min(open_p, close_p) * (1 - abs(random.gauss(0, 0.0006)))
        return CandleInput(
            symbol=symbol,
            ts=datetime.now(timezone.utc),
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=abs(random.gauss(1_000_000, 200_000)),
        ).model_dump()


class Brain:
    """Observes market data, runs strategies, stores forecasts, and reflects."""

    def __init__(
        self,
        database: Database = db,
        mem: Memory = memory,
        strategy: Optional[Strategy] = None,
        coordinator: Optional[StrategyCoordinator] = None,
        events: EventLog = event_log,
        colony: ColonySim = colony,
    ) -> None:
        self.db = database
        self.memory = mem
        self.events = events
        self.colony = colony
        self.data_feed = DataFeed(database=database)
        self.strategy = strategy or BinaryForecastStrategy(
            "binary_forecast",
            {"confidence_threshold": settings.confidence_threshold},
        )
        self.coordinator = coordinator or StrategyCoordinator(
            database=database,
            llm=hybrid_llm if should_use_hybrid() else llm,
            mem=mem,
        )
        self.reasoning_audit = ReasoningAudit(database=database)
        self.self_analysis = SelfAnalysisEngine(database=database, strategies=[self.strategy])
        self.consciousness_reporter = ConsciousnessReporter(self_analysis=self.self_analysis, llm_client=llm)
        self.horizon_s = settings.forecast_horizon_seconds
        self.strategy.db = database
        self.strategy.llm = hybrid_llm if should_use_hybrid() else llm
        self._running = False
        self._last_cycle: Optional[datetime] = None
        self._primary_goal_id: Optional[int] = None
        self._last_databricks_export: Optional[date] = None
        self._last_diary_date: Optional[date] = None
        self._last_self_improve_at: Optional[datetime] = None

    async def seed(self) -> None:
        await self.data_feed._seed_real_history(self.db)
        self.data_feed.seed(self.db)
        # Ensure a persistent primary goal exists
        goals = self.db.list_goals(status="active", limit=1)
        if goals:
            self._primary_goal_id = goals[0]["id"]
        else:
            self._primary_goal_id = self.db.create_goal(
                "Autonomous market analysis",
                "Continuously observe markets, forecast direction, learn from outcomes, and improve.",
                priority=100,
            )
        self.events.emit("Synthetic market history seeded", category="setup", phase="idle")

    async def _resolve_old_forecasts(self) -> None:
        """Mark expired pending forecasts as win/loss/expire based on latest price."""
        now = datetime.now(timezone.utc)
        unresolved = self.db.unresolved_forecasts(before=now)
        for fc in unresolved:
            created = datetime.fromisoformat(fc["created_at"].replace("Z", "+00:00")) if isinstance(fc["created_at"], str) else fc["created_at"]
            if (now - created).total_seconds() < self.horizon_s:
                continue
            latest = self.db.latest_candles(fc["symbol"], limit=1)
            if not latest:
                self.db.resolve_forecast(fc["id"], "expired", False)
                continue
            current_price = latest[0]["close"]
            prior = self.db.latest_candles(fc["symbol"], limit=2)
            prior_price = prior[1]["close"] if len(prior) > 1 else current_price
            if fc["direction"] == "flat":
                correct = abs(current_price - prior_price) / prior_price < 0.0005 if prior_price else False
                outcome = "win" if correct else "loss"
            elif fc["direction"] == "up":
                correct = current_price > prior_price
                outcome = "win" if correct else "loss"
            else:  # down
                correct = current_price < prior_price
                outcome = "win" if correct else "loss"
            self.db.resolve_forecast(fc["id"], outcome, correct)
            try:
                entry_candles = self.db.candles_before(fc["symbol"], created, limit=50)
                entry_features = fc.get("features") or {}
                market_data = {"symbol": fc["symbol"], "candles": entry_candles, "features": entry_features}
                horizon_return = None
                if prior_price:
                    horizon_return = (current_price - prior_price) / prior_price
                episode = episode_from_forecast(
                    fc,
                    market_data,
                    actor=fc.get("model_used", "unknown"),
                    outcome={
                        "win": outcome == "win",
                        "correct": correct,
                        "resolved_at": now.isoformat(),
                        "horizon_return": horizon_return,
                        "outcome": outcome,
                    },
                )
                episode_id = await self.memory.store_episode(episode)
            except Exception as exc:
                episode_id = None
                self.events.emit(
                    f"Episode capture failed for forecast #{fc['id']}: {exc}",
                    category="brain",
                    phase="reflect",
                    symbol=fc["symbol"],
                    level="WARN",
                )
            try:
                reasoning_output = fc.get("reasoning_output") or {"decision": "HOLD", "confidence": 0.0}
                await self.reasoning_audit.record(
                    reasoning_output=reasoning_output,
                    forecast_outcome={
                        "win": outcome == "win",
                        "correct": correct,
                        "horizon_return": horizon_return,
                    },
                    forecast_id=fc["id"],
                )
            except Exception as exc:
                self.events.emit(
                    f"Reasoning audit failed for forecast #{fc['id']}: {exc}",
                    category="brain",
                    phase="reflect",
                    symbol=fc["symbol"],
                    level="WARN",
                )
            self.events.emit(
                f"Resolved forecast #{fc['id']} {fc['symbol']} {fc['direction'].upper()} as {outcome}",
                category="brain",
                phase="reflect",
                symbol=fc["symbol"],
                metadata={"forecast_id": fc["id"], "outcome": outcome, "correct": correct, "episode_id": episode_id},
            )

    async def observe(self, symbol: str) -> dict[str, Any]:
        self.colony.set_phase("observe", symbol=symbol, task=f"scanning {symbol}")
        self.events.emit(f"Observing market data for {symbol}", category="brain", phase="observe", symbol=symbol)

        # Only one instance should generate synthetic market data at a time.
        lock = DistributedLock(f"observe_{symbol}", database=self.db, ttl_seconds=10)
        if lock.acquire(timeout_seconds=2.0):
            try:
                candle = self.data_feed.next_candle(symbol)
                self.db.insert_candle(candle)
            finally:
                lock.release()
        else:
            self.events.emit(
                f"Observe locked by peer for {symbol}; using latest candles",
                category="brain",
                phase="observe",
                symbol=symbol,
            )

        candles = self.db.latest_candles(symbol, limit=20)
        features = analyzer.summarize(candles)
        return {"symbol": symbol, "candles": candles, "features": features}

    async def plan(self, market_data: dict[str, Any]) -> list[Trade]:
        symbol = market_data["symbol"]
        self.colony.set_phase("plan", symbol=symbol, task=f"forecasting {symbol}")
        self.events.emit(f"Planning forecast for {symbol}", category="brain", phase="plan", symbol=symbol)
        task_id = self.db.create_task(
            "forecast",
            {"symbol": symbol, "features": market_data.get("features")},
            goal_id=self._primary_goal_id,
            status="running",
        )
        market_data["task_id"] = task_id
        with timed(llm_inference_latency_seconds):
            reasoning_output = await self.coordinator.coordinate(market_data)
        market_data["reasoning_output"] = reasoning_output

        # Convert reasoning decision into the Trade representation the rest of
        # the brain expects, preserving compatibility with act().
        trades: list[Trade] = []
        decision = reasoning_output.get("decision", "HOLD")
        if decision == "BUY":
            trades.append(
                Trade(
                    symbol=symbol,
                    side="buy",
                    quantity=reasoning_output.get("size_multiplier", 1.0),
                    strategy_id="reasoning_engine",
                    confidence=reasoning_output.get("confidence", 50.0),
                )
            )
        elif decision == "SELL":
            trades.append(
                Trade(
                    symbol=symbol,
                    side="sell",
                    quantity=reasoning_output.get("size_multiplier", 1.0),
                    strategy_id="reasoning_engine",
                    confidence=reasoning_output.get("confidence", 50.0),
                )
            )
        return trades

    async def act(self, symbol: str, trades: list[Trade], market_data: dict[str, Any]) -> Optional[int]:
        forecast: Optional[ForecastResult] = market_data.get("last_forecast")
        task_id: Optional[int] = market_data.get("task_id")

        # If no strategy produced a ForecastResult (e.g. pure mean-reversion),
        # synthesize one from the proposed trades.
        if forecast is None and trades:
            side = trades[0].side
            direction = Direction.UP if side == "buy" else (Direction.DOWN if side == "sell" else Direction.FLAT)
            forecast = ForecastResult(
                symbol=symbol,
                direction=direction,
                confidence=trades[0].confidence,
                rationale=f"Synthesized from {trades[0].strategy_id}",
                model_used=trades[0].strategy_id,
            )

        if forecast is None:
            if task_id:
                self.db.update_task_status(task_id, "done", {"result": "no_forecast"})
            return None
        self.colony.set_phase("act", symbol=symbol, task=f"acting on {symbol}")
        side = trades[0].side if trades else "none"
        self.events.emit(
            f"Acting on {symbol}: {forecast.direction.value} ({forecast.confidence:.1f}%) → {side}",
            category="brain",
            phase="act",
            symbol=symbol,
            metadata={"direction": forecast.direction.value, "confidence": forecast.confidence, "side": side},
        )
        reasoning_output = market_data.get("reasoning_output")
        forecast_id = self.db.insert_forecast(
            {
                "task_id": task_id,
                "symbol": symbol,
                "horizon_s": self.horizon_s,
                "direction": forecast.direction.value,
                "confidence": forecast.confidence,
                "rationale": forecast.rationale,
                "features": market_data.get("features"),
                "model_used": forecast.model_used,
                "reasoning_output": reasoning_output,
            }
        )
        if task_id:
            self.db.update_task_status(task_id, "done", {"forecast_id": forecast_id, "result": "forecast_created"})
        return forecast_id

    async def reflect(self, symbol: str, forecast: Optional[ForecastResult], market_data: dict[str, Any]) -> None:
        """Summarize recent activity, score it, and store a learning."""
        self.colony.set_phase("reflect", symbol=symbol, task=f"reflecting on {symbol}")
        recent = self.db.recent_forecasts(symbol, limit=10)
        wins = sum(1 for f in recent if f.get("outcome") == "win")
        losses = sum(1 for f in recent if f.get("outcome") == "loss")
        pending = sum(1 for f in recent if f.get("outcome") == "pending")
        features = market_data.get("features", {})
        score = 0.5
        critique_parts: list[str] = []
        if forecast:
            if forecast.confidence > 80 and recent and recent[0].get("outcome") == "loss":
                score = 0.2
                critique_parts.append("high confidence preceded a loss; consider widening uncertainty")
            elif wins > losses:
                score = 0.8
                critique_parts.append("recent forecasts outperform losses")
            else:
                critique_parts.append("performance is mixed; more data needed")
        critique = " ".join(critique_parts) or "No strong critique."

        summary = (
            f"{symbol}: processed cycle. Latest forecast: "
            f"{forecast.direction.value if forecast else 'none'} "
            f"({forecast.confidence if forecast else 0:.1f}% confidence). "
            f"Recent outcomes: {wins}W/{losses}L/{pending}P. "
            f"RSI={features.get('rsi', 0):.1f} vol={features.get('volatility', 0):.4f}"
        )

        task_id = self.db.create_task(
            "reflect",
            {"symbol": symbol, "summary": summary},
            goal_id=self._primary_goal_id,
            status="running",
        )
        self.db.insert_reflection({"task_id": task_id, "critique": critique, "score": score})
        await self.memory.remember(
            summary=summary,
            detail=str({"symbol": symbol, "features": features, "recent": recent[:3]}),
            task_id=task_id,
            tags=["cycle", "forecast", symbol.lower()],
        )
        self.db.update_task_status(task_id, "done", {"score": score})
        self.events.emit(summary, category="brain", phase="reflect", symbol=symbol)

    async def _record_hip4_predictions(self) -> list[dict[str, Any]]:
        """Fetch active HIP-4 daily binaries and store/update predictions.

        Uses the market-implied probability as confidence.  A prediction is
        updated in place for the same (underlying, expiry) pair, so only the
        latest daily prediction is kept per asset.
        """
        try:
            binaries = await get_daily_binaries()
        except Exception as exc:
            self.events.emit(f"HIP-4 prediction fetch failed: {exc}", category="hip4", phase="plan", level="WARN")
            return []

        recorded: list[dict[str, Any]] = []
        for b in binaries:
            probability = float(b["implied_probability"])
            if probability > 0.5:
                direction = "up"
                confidence = probability * 100.0
            elif probability < 0.5:
                direction = "down"
                confidence = (1.0 - probability) * 100.0
            else:
                direction = "flat"
                confidence = 50.0

            rationale = (
                f"HIP-4 daily binary: {b['underlying']} above {b['target_price']} "
                f"by {b['expiry_str']}; market-implied Yes probability {probability:.2%}"
            )
            prediction = {
                "underlying": b["underlying"],
                "outcome_id": b["outcome_id"],
                "expiry": b["expiry_dt"],
                "target_price": b["target_price"],
                "yes_price": b["yes_price"],
                "no_price": b["no_price"],
                "implied_probability": probability,
                "direction": direction,
                "confidence": confidence,
                "rationale": rationale,
            }
            try:
                self.db.insert_hip4_prediction(prediction)
                recorded.append(prediction)
                self.events.emit(
                    f"HIP-4 prediction {b['underlying']}: {direction.upper()} ({confidence:.1f}%) — {rationale}",
                    category="hip4",
                    phase="act",
                    symbol=b["underlying"],
                    metadata={"outcome_id": b["outcome_id"], "probability": probability, "target": b["target_price"]},
                )
            except Exception as exc:
                self.events.emit(f"HIP-4 prediction insert failed for {b['underlying']}: {exc}", category="hip4", phase="act", level="WARN")

        return recorded

    async def _maybe_export_to_databricks(self) -> None:
        """Push a daily snapshot to Databricks for external dashboards.

        Refreshes the system report first so the published snapshot always
        includes today's Borg health summary.
        """
        if not settings.databricks_enabled:
            return
        now = datetime.now(timezone.utc)
        if now.hour < 7 or self._last_databricks_export == now.date():
            return
        try:
            report_engine.generate_system_report(report_date=now.date())
        except Exception as exc:
            self.events.emit(f"System report generation failed: {exc}", category="databricks", phase="act", level="WARN")
        try:
            result = await export_all(database=self.db, events=self.events)
            if result.get("status") == "ok":
                self._last_databricks_export = now.date()
        except Exception as exc:
            self.events.emit(f"Databricks daily export failed: {exc}", category="databricks", phase="act", level="WARN")

    async def _maybe_paper_trade_hip4(self, predictions: list[dict[str, Any]]) -> None:
        """Open one daily paper trade on the strongest HIP-4 crypto option and settle any matured ones."""
        try:
            settled = await settle_open_paper_trades(database=self.db)
            for s in settled:
                self.events.emit(
                    f"HIP-4 paper trade #{s['id']} {s['underlying']} settled {s['outcome'].upper()} PnL {s['pnl']:.4f}",
                    category="hip4",
                    phase="reflect",
                    symbol=s["underlying"],
                    metadata=s,
                )
        except Exception as exc:
            self.events.emit(f"HIP-4 paper-trade settlement failed: {exc}", category="hip4", phase="reflect", level="WARN")

        try:
            trade = await create_daily_paper_trade(predictions, database=self.db)
            if trade:
                self.events.emit(
                    f"HIP-4 paper trade opened: {trade['underlying']} {trade['side']} {trade['direction'].upper()} "
                    f"stake ${trade['stake']:.2f} @ {trade['target_price']}",
                    category="hip4",
                    phase="act",
                    symbol=trade["underlying"],
                    metadata=trade,
                )
        except Exception as exc:
            self.events.emit(f"HIP-4 paper-trade creation failed: {exc}", category="hip4", phase="act", level="WARN")

    async def _process_symbol(self, symbol: str) -> str:
        """Run observe → plan → act → reflect for a single symbol."""
        market_data = await self.observe(symbol)
        trades = await self.plan(market_data)
        forecast = market_data.get("last_forecast")
        await self.act(symbol, trades, market_data)
        await self.reflect(symbol, forecast, market_data)
        side = trades[0].side if trades else "none"
        if forecast:
            forecasts_generated.labels(
                symbol=forecast.symbol, direction=forecast.direction.value
            ).inc()
        return f"{symbol}:{forecast.direction.value if forecast else '?'}/{side}"

    async def _maybe_self_improve(self) -> None:
        """Trigger self-improvement analysis once per configured interval."""
        now = datetime.now(timezone.utc)
        if self._last_self_improve_at and (now - self._last_self_improve_at).total_seconds() < settings.self_improve_interval_seconds:
            return
        try:
            proposal = await self_improve.analyze()
            self._last_self_improve_at = now
            if proposal and proposal.get("status") == "ok":
                self.events.emit(
                    f"Self-improvement proposed: {proposal.get('module')} ({proposal.get('version')})",
                    category="self_improve",
                    phase="reflect",
                    metadata=proposal,
                )
        except Exception as exc:
            self.events.emit(f"Self-improvement analysis failed: {exc}", category="self_improve", phase="reflect", level="WARN")

    async def cycle(self) -> dict[str, Any]:
        record_resource_usage()
        cycle_id = self.db.start_cycle()
        self.events.emit("Brain cycle started", category="brain", phase="idle")
        try:
            with timed(brain_cycle_duration_seconds):
                if monitor.should_throttle():
                    msg = "throttled by resource monitor"
                    self.events.emit(msg, category="system", phase="monitor", level="WARN")
                    self.db.finish_cycle(cycle_id, msg, "throttled")
                    brain_cycles_total.labels(status="throttled").inc()
                    return {"status": "throttled"}

                await self._resolve_old_forecasts()
                await self.data_feed.refresh_real_data(self.db)

                concurrency = max(1, settings.brain_concurrent_symbols)
                semaphore = asyncio.Semaphore(concurrency)

                async def _run_symbol(symbol: str) -> str:
                    async with semaphore:
                        return await self._process_symbol(symbol)

                summary_parts = await asyncio.gather(*[_run_symbol(symbol) for symbol in settings.symbol_list])

                recorded = await self._record_hip4_predictions()
                await self._maybe_paper_trade_hip4(recorded)
                await self._maybe_export_to_databricks()
                await self._maybe_self_improve()

            self._last_cycle = datetime.now(timezone.utc)
            summary = "; ".join(summary_parts)
            self.events.emit(f"Cycle complete: {summary}", category="brain", phase="idle")
            self.db.finish_cycle(cycle_id, summary, "ok")
            brain_cycles_total.labels(status="ok").inc()
            return {"status": "ok", "summary": summary}
        except Exception as exc:
            self.events.emit(f"Cycle error: {exc}", category="system", phase="idle", level="ERROR")
            self.db.finish_cycle(cycle_id, str(exc), "error")
            errors_total.labels(component="brain").inc()
            brain_cycles_total.labels(status="error").inc()
            return {"status": "error", "error": str(exc)}

    async def run(self) -> None:
        self._running = True
        while self._running:
            await self.cycle()
            await monitor.sleep_adaptive(settings.brain_interval_seconds)

    async def run_reporting_loop(self, interval_seconds: int = 60) -> None:
        """Generate daily reports at 09:00 and weekly reports on Monday 09:00."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                if now.hour == 9 and now.minute == 0:
                    today = now.date().isoformat()
                    if not self.db.get_consciousness_report("daily", today):
                        report = await self.consciousness_reporter.generate_daily_report()
                        score = ConsciousnessScore.from_system_state({
                            "reasoning_accuracy": 1.0 - (await self.reasoning_audit.get_calibration_report(window_days=30)).get("mean_calibration_error", 0.0),
                            "learning_updates_count": len([a for a in self.db.recent_audit(limit=100) if a.get("action") == "learning_update"]),
                            "report_count": len(self.db.query_consciousness_reports(limit=1000)),
                            "performance_consistency": max(0.0, min(1.0, self.self_analysis.analyze_performance("weekly").get("sharpe", 0.0))),
                        })
                        self.db.insert_consciousness_report(
                            {
                                "period": "daily",
                                "report_date": today,
                                "report_text": report,
                                "score": score,
                                "metadata": {"generated_at": now.isoformat()},
                            }
                        )
                        self.events.emit("Daily consciousness report generated", category="consciousness", phase="reflect")

                    if now.weekday() == 0:
                        if not self.db.get_consciousness_report("weekly", today):
                            report = await self.consciousness_reporter.generate_weekly_report()
                            self.db.insert_consciousness_report(
                                {
                                    "period": "weekly",
                                    "report_date": today,
                                    "report_text": report,
                                    "score": None,
                                    "metadata": {"generated_at": now.isoformat()},
                                }
                            )
                            self.events.emit("Weekly consciousness report generated", category="consciousness", phase="reflect")

                    # Daily diary at configured time (default 23:59 UTC).
                    if (
                        settings.diary_enabled
                        and now.hour == settings.diary_hour
                        and now.minute == settings.diary_minute
                        and self._last_diary_date != now.date()
                    ):
                        try:
                            from borg.modules.hyperlong_client import fetch_all_symbols

                            hyperlong_data = await fetch_all_symbols()
                            diary_path = diary_writer.write_daily_diary(
                                now.date(), hyperlong_data=hyperlong_data
                            )
                            self._last_diary_date = now.date()
                            self.events.emit(
                                f"Daily diary written to {diary_path}",
                                category="diary",
                                phase="reflect",
                                metadata={"path": str(diary_path)},
                            )
                        except Exception as exc:
                            self.events.emit(f"Daily diary generation failed: {exc}", category="diary", phase="reflect", level="WARN")

                await asyncio.sleep(interval_seconds)
            except Exception as exc:
                logging.getLogger("borg").error("Reporting loop error: %s", exc)
                await asyncio.sleep(interval_seconds)

    async def run_learning_loop(
        self,
        dry_run: bool = False,
        validate: bool = True,
        interval_seconds: Optional[int] = None,
    ) -> None:
        """Daily: backtest last 90 days, propose weights, validate, deploy."""
        from borg.versioning import versioning

        interval = interval_seconds or 86400
        while self._running:
            try:
                logger = logging.getLogger("borg")
                logger.info("Starting daily learning loop")

                train_end = datetime.now(timezone.utc)
                train_start = train_end - timedelta(days=90)
                val_start = train_end
                val_end = val_start + timedelta(days=10)

                coordinator = self.coordinator
                engine = BacktestEngine(coordinator=coordinator, database=self.db, memory=self.memory)
                learner = LearningEngine(coordinator=coordinator)

                train_results = await engine.run_backtest(train_start, train_end, record_episodes=True)
                logger.info(
                    "Training backtest: %d trades, %.1f%% win, Sharpe %.2f",
                    train_results["trade_count"],
                    train_results["win_rate"] * 100,
                    train_results["sharpe"],
                )

                current_weights = coordinator.get_weights()
                proposed = learner.propose_weight_updates(train_results, current_weights)
                logger.info("Proposed updates: %s", proposed)

                if dry_run:
                    logger.info("Learning dry-run complete")
                else:
                    coordinator.apply_weights(proposed)

                    if validate:
                        val_results = await engine.run_backtest(val_start, val_end, record_episodes=False)
                        logger.info(
                            "Validation backtest: %d trades, %.1f%% win, Sharpe %.2f",
                            val_results["trade_count"],
                            val_results["win_rate"] * 100,
                            val_results["sharpe"],
                        )
                        safe = learner.validate_updates(proposed, train_results, val_results)
                        if safe:
                            await versioning.record_learning_update(
                                updates={f"{k[0]}::{k[1]}": v for k, v in proposed.items()},
                                performance_before=train_results,
                                performance_after=val_results,
                            )
                            logger.info("Learning updates validated and deployed")
                        else:
                            logger.warning("Learning updates failed validation; rolling back")
                            coordinator.apply_weights(current_weights)
                    else:
                        await versioning.record_learning_update(
                            updates={f"{k[0]}::{k[1]}": v for k, v in proposed.items()},
                            performance_before=train_results,
                            performance_after=train_results,
                        )
                        logger.info("Learning updates deployed without validation")
            except Exception as exc:
                logging.getLogger("borg").error("Learning loop error: %s", exc)

            if interval_seconds == 0:
                break
            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False

    def last_cycle(self) -> Optional[datetime]:
        return self._last_cycle


brain = Brain()
