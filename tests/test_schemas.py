"""Validation tests for Pydantic schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from borg.schemas import CandleInput, ForecastInput


def test_candle_ok() -> None:
    candle = CandleInput(
        symbol="EURUSD",
        ts="2025-01-15T12:30:00Z",
        open=1.0840,
        high=1.0860,
        low=1.0820,
        close=1.0850,
        volume=1_500_000,
    )
    assert candle.symbol == "EURUSD"
    assert candle.high >= candle.low


def test_candle_bad_ohlc() -> None:
    with pytest.raises(ValidationError):
        CandleInput(
            symbol="EURUSD",
            ts="2025-01-15T12:30:00Z",
            open=1.0840,
            high=1.0800,  # high < low
            low=1.0820,
            close=1.0850,
            volume=1_500_000,
        )


def test_candle_negative_volume() -> None:
    with pytest.raises(ValidationError):
        CandleInput(
            symbol="EURUSD",
            ts="2025-01-15T12:30:00Z",
            open=1.0840,
            high=1.0860,
            low=1.0820,
            close=1.0850,
            volume=-1,
        )


def test_forecast_overconfident() -> None:
    with pytest.raises(ValidationError):
        ForecastInput(symbol="EURUSD", direction="up", confidence=99, rationale="sure thing")


def test_forecast_direction_normalization() -> None:
    f = ForecastInput(symbol="eurusd", direction="down", confidence=75)
    assert f.direction.value == "down"
