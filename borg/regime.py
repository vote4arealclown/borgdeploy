"""Market regime detection based on simple trend and volatility rules."""
from __future__ import annotations

from typing import Any, Optional


def _sma(values: list[float], period: int) -> Optional[float]:
    """Simple moving average over the most recent `period` values.

    Values are expected newest-first (latest close at index 0).
    """
    if len(values) < period:
        return None
    return sum(values[:period]) / period


def _atr(candles: list[dict[str, Any]], period: int = 14) -> Optional[float]:
    """Average true range using high/low/close."""
    if len(candles) < period + 1:
        return None
    # Candles are newest-first; work with reversed copy for chronological order.
    chronological = list(reversed(candles))
    trs: list[float] = []
    for i in range(1, len(chronological)):
        high = float(chronological[i]["high"])
        low = float(chronological[i]["low"])
        prev_close = float(chronological[i - 1]["close"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        if len(trs) >= period:
            break
    return sum(trs) / len(trs) if trs else None


def _volatility_band(candles: list[dict[str, Any]]) -> tuple[float, float]:
    """Return (mean_close, normalized_atr) for the available window."""
    closes = [float(c["close"]) for c in candles if "close" in c]
    if not closes:
        return 0.0, 0.0
    mean_close = sum(closes) / len(closes)
    atr = _atr(candles)
    normalized = (atr / mean_close) if atr and mean_close else 0.0
    return mean_close, normalized


class RegimeDetector:
    """Detect market regime from OHLCV candles.

    Returns labels such as ``bull_low_vol``, ``bear_high_vol``,
    ``sideways_normal_vol``.  The detector is deterministic and stateless so
    it works with or without an LLM available.
    """

    def __init__(self) -> None:
        self.trend_threshold = 0.02
        self.high_vol_threshold = 0.015
        self.low_vol_threshold = 0.005

    def detect(self, candles: list[dict[str, Any]]) -> str:
        """Return a regime label for the supplied candle window."""
        closes = [float(c["close"]) for c in candles if "close" in c]
        if len(closes) < 20:
            return "unknown_normal_vol"

        sma_20 = _sma(closes, 20)
        sma_50 = _sma(closes, 50) if len(closes) >= 50 else sma_20

        if sma_20 is None or sma_50 is None:
            return "unknown_normal_vol"

        if sma_20 > sma_50 * (1 + self.trend_threshold):
            trend = "bull"
        elif sma_20 < sma_50 * (1 - self.trend_threshold):
            trend = "bear"
        else:
            trend = "sideways"

        _, normalized_atr = _volatility_band(candles)
        if normalized_atr > self.high_vol_threshold:
            vol = "high_vol"
        elif normalized_atr < self.low_vol_threshold:
            vol = "low_vol"
        else:
            vol = "normal_vol"

        return f"{trend}_{vol}"


def detect_regime(candles: list[dict[str, Any]]) -> str:
    """Convenience function using the default detector."""
    return RegimeDetector().detect(candles)
