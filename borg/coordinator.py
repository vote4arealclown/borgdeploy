"""Strategy coordinator: load and orchestrate multiple trading strategies."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from borg.config import PROJECT_ROOT
from borg.db import Database, db
from borg.llm import LLMClient
from borg.memory import Memory, memory
from borg.reasoning import ReasoningEngine
from borg.reflection import ReflectionEngine
from borg.regime import detect_regime
from borg.safety import ActionKind, SafetyGate, safety
from borg.strategies.base import Strategy, Trade
from borg.strategies.binary_forecast import BinaryForecastStrategy
from borg.strategies.consensus import ConsensusStrategy
from borg.strategies.dca import DCAStrategy
from borg.strategies.grid_trading import GridTradingStrategy
from borg.strategies.market_making import MarketMakingStrategy
from borg.strategies.mean_reversion import MeanReversionStrategy
from borg.strategies.momentum import MomentumStrategy
from borg.strategies.pairs_trading import PairsTradingStrategy
from borg.strategies.stat_arb import StatArbStrategy
from borg.strategies.trend_following import TrendFollowingStrategy
from borg.strategies.twap import TWAPStrategy
from borg.strategies.vwap import VWAPStrategy


class StrategyCoordinator:
    """Load strategies from YAML config and execute them against market data."""

    def __init__(
        self,
        database: Database = db,
        llm: Optional[LLMClient] = None,
        config_file: Optional[Path] = None,
        mem: Optional[Memory] = None,
    ) -> None:
        self.db = database
        self.llm = llm
        self.memory = mem or memory
        self.config_file = config_file or (PROJECT_ROOT / "config" / "strategies.yaml")
        self.strategies: dict[str, Strategy] = {}
        self._weights: dict[tuple[str, str], float] = {}
        self._reflection = ReflectionEngine(self.memory)
        self._reasoning = ReasoningEngine(llm_client=self.llm, memory=self.memory)
        self._safety = safety
        self._load_strategies()

    def _load_strategies(self) -> None:
        """Instantiate strategies from the YAML file."""
        config: dict[str, Any] = {"strategies": [], "meta_strategies": []}
        if self.config_file.exists():
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    config.update(loaded)

        # Primitive strategies.
        for s_config in config.get("strategies", []):
            s_type = s_config.get("type")
            s_name = s_config.get("name")
            s_params = s_config.get("params", {})
            if s_config.get("enabled", True) is False:
                s_params["enabled"] = False

            strategy: Optional[Strategy] = None
            if s_type == "binary_forecast":
                strategy = BinaryForecastStrategy(s_name, s_params)
            elif s_type == "mean_reversion":
                strategy = MeanReversionStrategy(s_name, s_params)
            elif s_type == "momentum":
                strategy = MomentumStrategy(s_name, s_params)
            elif s_type == "pairs_trading":
                strategy = PairsTradingStrategy(s_name, s_params)
            elif s_type == "trend_following":
                strategy = TrendFollowingStrategy(s_name, s_params)
            elif s_type == "stat_arb":
                strategy = StatArbStrategy(s_name, s_params)
            elif s_type == "vwap":
                strategy = VWAPStrategy(s_name, s_params)
            elif s_type == "twap":
                strategy = TWAPStrategy(s_name, s_params)
            elif s_type == "grid_trading":
                strategy = GridTradingStrategy(s_name, s_params)
            elif s_type == "market_making":
                strategy = MarketMakingStrategy(s_name, s_params)
            elif s_type == "dca":
                strategy = DCAStrategy(s_name, s_params)

            if strategy is not None:
                self.strategies[s_name] = strategy

        # Meta-strategies.
        for meta_config in config.get("meta_strategies", []):
            meta_name = meta_config.get("name")
            sub_names = meta_config.get("strategies", [])
            sub_strategies: list[Strategy] = []
            for name in sub_names:
                if name in self.strategies:
                    sub_strategies.append(self.strategies[name])

            if sub_strategies:
                self.strategies[meta_name] = ConsensusStrategy(
                    meta_name,
                    meta_config.get("params", {}),
                    sub_strategies,
                )

    async def execute(self, market_data: dict[str, Any]) -> list[Trade]:
        """Run active strategies for the given symbol and return aggregate trades."""
        all_trades: list[Trade] = []

        for strategy in self.strategies.values():
            if not strategy.config.get("enabled", True):
                continue
            strategy.db = self.db
            strategy.llm = self.llm
            try:
                trades = await strategy.analyze(market_data)
                all_trades.extend(trades)
            except Exception as exc:
                from borg.events import event_log

                event_log.emit(
                    f"Strategy {strategy.name} failed: {exc}",
                    category="strategy",
                    phase="plan",
                    symbol=market_data.get("symbol"),
                )

        return self._apply_risk_limits(all_trades)

    async def execute_with_reflection(
        self,
        market_data: dict[str, Any],
        regime: Optional[str] = None,
    ) -> list[Trade]:
        """Run strategies, weight by regime efficacy, and apply reflection."""
        detected_regime = regime or detect_regime(market_data.get("candles", []))
        all_trades = await self.execute(market_data)
        if not all_trades:
            return []

        adjusted: list[Trade] = []
        for trade in all_trades:
            strategy_name = trade.strategy_id.split()[0] if trade.strategy_id else "unknown"
            efficacy = await self.memory.get_strategy_efficacy(
                strategy_name=strategy_name,
                regime=detected_regime,
                window_days=30,
            )
            weight_key = (strategy_name, detected_regime)
            base_weight = self._weights.get(weight_key, 1.0)

            if efficacy.sample_size >= 5:
                efficacy_weight = 1.0 + (efficacy.win_rate - 0.5)
                efficacy_weight = min(2.0, max(0.5, efficacy_weight))
            else:
                efficacy_weight = 1.0

            total_weight = base_weight * efficacy_weight
            reflection = await self._reflection.reflect_on_signal(
                {
                    "strategy": strategy_name,
                    "trigger": f"direction={trade.side} confidence={trade.confidence:.1f}",
                    "confidence": trade.confidence,
                },
                detected_regime,
            )
            reflection_adj = reflection.get("confidence_adj", 1.0)

            trade.confidence = min(95.0, trade.confidence * total_weight * reflection_adj)
            adjusted.append(trade)

        return self._apply_risk_limits(adjusted)

    async def coordinate(
        self,
        market_data: dict[str, Any],
        regime: Optional[str] = None,
    ) -> dict[str, Any]:
        """Strategies -> reflection -> reasoning -> execution decision."""
        detected_regime = regime or detect_regime(market_data.get("candles", []))

        # Step 1: collect raw signals.
        raw_trades = await self.execute(market_data)
        signals: list[dict[str, Any]] = []
        for trade in raw_trades:
            strategy_name = trade.strategy_id.split()[0] if trade.strategy_id else "unknown"
            signals.append(
                {
                    "strategy": strategy_name,
                    "action": trade.side.upper(),
                    "confidence": trade.confidence,
                    "trigger": f"direction={trade.side} confidence={trade.confidence:.1f}",
                }
            )

        if not signals:
            return {"decision": "HOLD", "confidence": 0.0, "reasoning": "No strategy signals"}

        # Step 2: reason.
        reasoning_output = await self._reasoning.reason(
            market_data=market_data,
            signals=signals,
            regime=detected_regime,
        )

        # Step 3: safety check.
        decision = reasoning_output.get("decision", "HOLD")
        safety_result = self._safety.check(
            ActionKind.FORECAST,
            {"decision": decision, "confidence": reasoning_output.get("confidence")},
        )
        if not safety_result.get("approved"):
            reasoning_output["decision"] = "HOLD"
            reasoning_output["reasoning"] = "Blocked by safety gate: " + str(
                safety_result.get("detail", {})
            )

        # Step 4: enrich for audit.
        reasoning_output["regime"] = detected_regime
        reasoning_output["signals_input"] = signals
        reasoning_output["timestamp"] = datetime.now(timezone.utc).isoformat()
        return reasoning_output

    def get_weights(self) -> dict[tuple[str, str], float]:
        """Return a copy of current strategy/regime weights."""
        return dict(self._weights)

    def apply_weights(self, weights: dict[tuple[str, str], float]) -> None:
        """Apply new strategy/regime weights."""
        self._weights = {k: float(v) for k, v in weights.items()}
        for strategy in self.strategies.values():
            strategy.clear_efficacy_cache()

    def reset_weights(self) -> None:
        """Clear all learned weights."""
        self._weights = {}
        for strategy in self.strategies.values():
            strategy.clear_efficacy_cache()

    def _apply_risk_limits(self, trades: list[Trade]) -> list[Trade]:
        """Apply conservative position sizing across all proposed trades."""
        if not trades:
            return trades

        # Simple sizing: cap total notional exposure per symbol.
        max_notional = 1_000.0
        by_symbol: dict[str, list[Trade]] = {}
        for trade in trades:
            by_symbol.setdefault(trade.symbol, []).append(trade)

        result: list[Trade] = []
        for symbol, symbol_trades in by_symbol.items():
            total_qty = sum(t.quantity for t in symbol_trades)
            if total_qty <= 0:
                continue
            price = symbol_trades[0].entry_price or 1.0
            scale = min(1.0, max_notional / (total_qty * price))
            for trade in symbol_trades:
                if scale < 1.0:
                    trade.quantity = round(trade.quantity * scale, 4)
                result.append(trade)

        return result

    def list_strategy_names(self) -> list[str]:
        """Return names of all loaded strategies."""
        return list(self.strategies.keys())
