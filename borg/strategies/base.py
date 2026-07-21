"""Base strategy abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional



@dataclass
class Trade:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float = 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_id: str = ""
    confidence: float = 50.0


class Strategy(ABC):
    """Abstract trading strategy."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.db: Any = None
        self.llm: Any = None
        self._efficacy_cache: dict[str, Any] = {}

    @abstractmethod
    async def analyze(self, market_data: dict[str, Any]) -> list[Trade]:
        """Return proposed trades for the given market snapshot."""
        raise NotImplementedError

    async def get_efficacy(self, regime: str, memory: Any) -> Any:
        """Query memory for this strategy's performance in the given regime."""
        if regime not in self._efficacy_cache:
            self._efficacy_cache[regime] = await memory.get_strategy_efficacy(
                strategy_name=self.name,
                regime=regime,
                window_days=30,
            )
        return self._efficacy_cache[regime]

    def clear_efficacy_cache(self) -> None:
        """Reset cached efficacy, used after learning updates."""
        self._efficacy_cache = {}

    def risk_metrics(self) -> dict[str, Any]:
        return {
            "max_drawdown_pct": 5.0,
            "max_position_size_usd": 100.0,
            "max_leverage": 1.0,
        }

    def _format_summary(self, candles: list[dict[str, Any]]) -> str:
        if not candles:
            return "no data"
        latest = candles[0]
        prev = candles[1] if len(candles) > 1 else latest
        change = latest["close"] - prev["close"]
        pct = (change / prev["close"] * 100) if prev["close"] else 0.0
        return (
            f"{latest['symbol']} close={latest['close']:.5f} change={change:+.5f} ({pct:+.3f}%) "
            f"volume={latest['volume']:.0f} high={latest['high']:.5f} low={latest['low']:.5f}"
        )
