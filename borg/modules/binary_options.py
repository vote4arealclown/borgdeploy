"""Binary-options market analysis helpers."""
from __future__ import annotations

from typing import Any


class BinaryOptionsAnalyzer:
    """Lightweight technical helpers for binary-option directional signals."""

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def rsi(candles: list[dict[str, Any]], period: int = 14) -> float:
        """Simple RSI approximation using closing prices."""
        if len(candles) < period + 1:
            return 50.0
        closes = [BinaryOptionsAnalyzer._as_float(c["close"]) for c in candles[: period + 1]][::-1]
        gains = sum(max(0, closes[i] - closes[i - 1]) for i in range(1, len(closes)))
        losses = sum(max(0, closes[i - 1] - closes[i]) for i in range(1, len(closes)))
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def sma(candles: list[dict[str, Any]], period: int = 5) -> float:
        if len(candles) < period:
            return BinaryOptionsAnalyzer._as_float(candles[0]["close"]) if candles else 0.0
        return sum(BinaryOptionsAnalyzer._as_float(c["close"]) for c in candles[:period]) / period

    @staticmethod
    def volatility(candles: list[dict[str, Any]], period: int = 10) -> float:
        if len(candles) < 2:
            return 0.0
        sample = candles[:period]
        closes = [BinaryOptionsAnalyzer._as_float(c["close"]) for c in sample]
        mean = sum(closes) / len(closes)
        variance = sum((c - mean) ** 2 for c in closes) / len(closes)
        return (variance ** 0.5) / mean if mean else 0.0

    @staticmethod
    def summarize(candles: list[dict[str, Any]]) -> dict[str, Any]:
        if not candles:
            return {}
        latest = candles[0]
        prev = candles[1] if len(candles) > 1 else latest
        latest_close = BinaryOptionsAnalyzer._as_float(latest["close"])
        prev_close = BinaryOptionsAnalyzer._as_float(prev["close"])
        change = latest_close - prev_close
        pct = (change / prev_close * 100.0) if prev_close else 0.0
        return {
            "symbol": latest.get("symbol"),
            "close": latest_close,
            "change": change,
            "change_pct": pct,
            "volume": BinaryOptionsAnalyzer._as_float(latest.get("volume", 0)),
            "rsi": BinaryOptionsAnalyzer.rsi(candles),
            "sma_5": BinaryOptionsAnalyzer.sma(candles, 5),
            "volatility": BinaryOptionsAnalyzer.volatility(candles),
        }


analyzer = BinaryOptionsAnalyzer()
