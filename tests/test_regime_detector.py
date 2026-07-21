"""Tests for market regime detection."""
from __future__ import annotations

import pytest

from borg.regime import RegimeDetector, detect_regime


def _candles(trend: str, volatility: str, count: int = 60) -> list[dict]:
    """Generate a synthetic candle window with the requested trend/volatility."""
    import random

    rng = random.Random(f"{trend}_{volatility}_{count}")
    base = 100.0
    candles: list[dict] = []
    for i in range(count):
        if trend == "up":
            drift = i * 0.003
        elif trend == "down":
            drift = -i * 0.003
        else:
            drift = 0.0

        if volatility == "high":
            noise = rng.uniform(-0.02, 0.02)
        elif volatility == "low":
            noise = rng.uniform(-0.002, 0.002)
        else:
            noise = rng.uniform(-0.008, 0.008)

        close = base * (1 + drift + noise)
        high = close * (1 + abs(rng.gauss(0, 0.005)))
        low = close * (1 - abs(rng.gauss(0, 0.005)))
        # Ensure OHLC ordering
        high, low = max(high, low), min(high, low)
        open_p = low + rng.random() * (high - low)
        candles.append(
            {
                "symbol": "TEST",
                "ts": f"2026-07-01T00:{i:02d}:00",
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1_000_000,
            }
        )
    # Newest first
    return list(reversed(candles))


def test_detect_regime_bull_low_vol() -> None:
    candles = _candles("up", "low")
    regime = detect_regime(candles)
    assert regime.startswith("bull_")


def test_detect_regime_bear_high_vol() -> None:
    candles = _candles("down", "high")
    regime = detect_regime(candles)
    assert regime.startswith("bear_")
    assert "high_vol" in regime


def test_detect_regime_sideways() -> None:
    candles = _candles("sideways", "normal")
    regime = detect_regime(candles)
    assert regime.startswith("sideways_")


def test_detect_regime_short_window_returns_unknown() -> None:
    candles = _candles("up", "low", count=5)
    regime = detect_regime(candles)
    assert regime.startswith("unknown_")


def test_regime_detector_stateless() -> None:
    detector = RegimeDetector()
    candles = _candles("up", "low")
    assert detector.detect(candles) == detect_regime(candles)
