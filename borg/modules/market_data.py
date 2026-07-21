"""Real market-data feed via the Hyperliquid public API.

Fetches 1m candles from Hyperliquid's UI API (which exposes native perps and
HIP-3 builder-deployed perps).  Native crypto symbols pass through as-is;
forex pairs are mapped to the builder-internal names the frontend uses
(e.g. EURUSD -> xyz:EUR).  The feed falls back gracefully if the API is down
or a symbol is unknown.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from borg.config import settings
from borg.schemas import CandleInput

logger = logging.getLogger(__name__)

# The frontend/private API exposes HIP-3 builder perps that are not present on
# the public docs endpoint.
HYPERLIQUID_API_URL = "https://api-ui.hyperliquid.xyz/info"

# Map Borg symbols to Hyperliquid coin names.  Defaults to identity so the
# config symbols list can be used directly (BTC, ETH, SOL, etc.).
HYPERLIQUID_COIN_MAP: dict[str, str] = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "XLM": "XLM",
    "XRP": "XRP",
    "BNB": "BNB",
    "HYPE": "HYPE",
    # HIP-3 builder forex (Unit/xyz).  Internal names are the base currency;
    # the quote is implicitly USD for EUR/GBP and JPY for xyz:JPY.
    "EURUSD": "xyz:EUR",
    "GBPUSD": "xyz:GBP",
    "USDJPY": "xyz:JPY",
}


def _period_to_ms(period: str) -> int:
    """Convert a simple period string to milliseconds."""
    value = int("".join(c for c in period if c.isdigit()) or "1")
    unit = "".join(c for c in period if c.isalpha()).lower()
    multipliers = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
    }
    return value * multipliers.get(unit, 86_400_000)


async def _fetch_coin_candles(
    client: httpx.AsyncClient,
    symbol: str,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> tuple[str, list[CandleInput]]:
    """Fetch candles for a single Hyperliquid coin."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    try:
        resp = await client.post(HYPERLIQUID_API_URL, json=payload, timeout=15.0)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        logger.warning("Hyperliquid fetch failed for %s (%s): %s", symbol, coin, exc)
        return symbol, []

    candles: list[CandleInput] = []
    for row in rows:
        try:
            ts_ms = int(row["t"])
            ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            open_p = float(row["o"])
            high_p = float(row["h"])
            low_p = float(row["l"])
            close_p = float(row["c"])
            volume = float(row["v"])
            if close_p <= 0.0 or any(v != v for v in (open_p, high_p, low_p, close_p, volume)):  # NaN check
                continue
            candles.append(
                CandleInput(
                    symbol=symbol,
                    ts=ts,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=volume,
                )
            )
        except Exception as exc:
            logger.debug("Skipping malformed Hyperliquid candle row for %s: %s", symbol, exc)
            continue

    return symbol, candles


async def fetch_latest_candles(
    symbols: Optional[list[str]] = None,
    period: str = "1d",
    interval: str = "1m",
) -> dict[str, list[CandleInput]]:
    """Fetch the latest real candles from Hyperliquid for the configured symbols.

    Returns an empty dict if the API call fails.  Duplicates are handled by the
    database's unique (symbol, ts) constraint.
    """
    symbols = symbols or settings.symbol_list
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - _period_to_ms(period)

    result: dict[str, list[CandleInput]] = {}
    async with httpx.AsyncClient() as client:
        tasks = [
            _fetch_coin_candles(
                client,
                symbol,
                HYPERLIQUID_COIN_MAP.get(symbol, symbol),
                interval,
                start_ms,
                end_ms,
            )
            for symbol in symbols
        ]
        for coro in asyncio.as_completed(tasks):
            symbol, candles = await coro
            if candles:
                result[symbol] = candles

    return result
