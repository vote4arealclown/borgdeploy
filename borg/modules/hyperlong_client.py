"""Client for the HyperLong local trading dashboard.

Fetches chart/indicator data from a HyperLong instance (e.g.
http://10.0.0.100:8080) and exposes it to Borg for reporting and enrichment.
"""
from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import quote

import httpx

from borg.config import settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://10.0.0.100:8080"
DEFAULT_TIMEOUT = 15.0


def _normalize_symbol(symbol: str) -> str:
    """Convert a Borg symbol into a HyperLong market pair if possible."""
    symbol = symbol.upper().strip()
    mapping = {
        "BTC": "BTC/USDC:USDC",
        "ETH": "ETH/USDC:USDC",
        "SOL": "SOL/USDC:USDC",
        "HYPE": "HYPE/USDC:USDC",
    }
    return mapping.get(symbol, symbol)


async def fetch_chart_data(
    symbol: str,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    """Fetch chart + indicator data for a symbol from HyperLong.

    Returns a dict with keys like price, ema100, ema200, bb_upper, bb_lower,
    rsi14, stoch_rsi, adx14, labels, markers, symbol.
    """
    base_url = (base_url or settings.hyperlong_base_url or DEFAULT_BASE_URL).rstrip("/")
    timeout = timeout or settings.hyperlong_timeout_seconds or DEFAULT_TIMEOUT
    pair = _normalize_symbol(symbol)
    url = f"{base_url}/api/chart?symbol={quote(pair, safe='')}"

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        data["borg_symbol"] = symbol
        data["hyperlong_symbol"] = pair
        return data
    except Exception as exc:
        logger.warning("HyperLong chart fetch failed for %s: %s", symbol, exc)
        return {"error": str(exc), "borg_symbol": symbol, "hyperlong_symbol": pair}
    finally:
        if own_client:
            await client.aclose()


async def fetch_all_symbols(
    symbols: Optional[list[str]] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """Fetch HyperLong data for all watched Borg symbols."""
    symbols = symbols or settings.symbol_list
    results: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        for symbol in symbols:
            results[symbol] = await fetch_chart_data(symbol, base_url=base_url, client=client)
    return results
