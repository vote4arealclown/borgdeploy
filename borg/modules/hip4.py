"""HIP-4 outcome market data from Hyperliquid.

HIP-4 introduces on-chain binary outcome markets.  This module fetches the
active daily price-binary markets for configured underlyings (BTC, ETH, SOL,
HYPE, etc.) and returns the market-implied probability plus strike/expiry data.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

HIP4_API_URL = "https://api.hyperliquid.xyz/info"

DEFAULT_UNDERLYINGS = ["BTC", "ETH", "SOL", "HYPE"]


def _parse_description(description: str) -> dict[str, Any]:
    """Parse a Hyperliquid HIP-4 description string into key/value pairs."""
    result: dict[str, Any] = {}
    for part in description.split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _parse_expiry(expiry_str: str) -> Optional[datetime]:
    """Parse '20260720-0600' into a UTC datetime."""
    try:
        return datetime.strptime(expiry_str, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
    except Exception:
        return None


async def fetch_outcome_meta(client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    """Fetch HIP-4 outcome metadata from Hyperliquid."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()
    try:
        resp = await client.post(HIP4_API_URL, json={"type": "outcomeMeta"}, timeout=15.0)
        resp.raise_for_status()
        return resp.json()
    finally:
        if own_client:
            await client.aclose()


async def fetch_all_mids(client: Optional[httpx.AsyncClient] = None) -> dict[str, float]:
    """Fetch all mid prices from Hyperliquid, including outcome tokens."""
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()
    try:
        resp = await client.post(HIP4_API_URL, json={"type": "allMids"}, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        return {str(k): float(v) for k, v in data.items()}
    finally:
        if own_client:
            await client.aclose()


async def get_daily_binaries(
    underlyings: Optional[list[str]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> list[dict[str, Any]]:
    """Return active daily price-binary markets for the requested underlyings.

    Each result contains:
        - underlying: coin name (BTC, ETH, ...)
        - outcome_id: HIP-4 outcome number
        - period: e.g. '1d'
        - expiry_str: raw expiry string from description
        - expiry_dt: parsed UTC datetime
        - target_price: strike price
        - yes_price: current market price of Yes token
        - no_price: current market price of No token
        - implied_probability: yes_price / (yes_price + no_price)
    """
    underlyings = underlyings or DEFAULT_UNDERLYINGS
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()
    try:
        meta, mids = await asyncio.gather(
            fetch_outcome_meta(client),
            fetch_all_mids(client),
        )
    finally:
        if own_client:
            await client.aclose()

    results: list[dict[str, Any]] = []
    for outcome in meta.get("outcomes", []):
        desc = outcome.get("description", "")
        parsed = _parse_description(desc)
        if parsed.get("class") != "priceBinary":
            continue
        if parsed.get("period") != "1d":
            continue
        underlying = parsed.get("underlying")
        if underlying not in underlyings:
            continue

        outcome_id = outcome["outcome"]
        yes_key = f"#{outcome_id}0"
        no_key = f"#{outcome_id}1"
        yes_price = mids.get(yes_key, 0.0)
        no_price = mids.get(no_key, 0.0)
        denom = yes_price + no_price
        implied_probability = yes_price / denom if denom > 0 else 0.0

        expiry_dt = _parse_expiry(parsed.get("expiry", ""))
        if expiry_dt is None:
            continue

        try:
            target_price = float(parsed.get("targetPrice", "0"))
        except Exception:
            target_price = 0.0

        results.append(
            {
                "underlying": underlying,
                "outcome_id": outcome_id,
                "period": parsed.get("period"),
                "expiry_str": parsed.get("expiry"),
                "expiry_dt": expiry_dt,
                "target_price": target_price,
                "yes_price": yes_price,
                "no_price": no_price,
                "implied_probability": implied_probability,
                "description": desc,
            }
        )

    return results


# Import asyncio at the bottom to avoid circular issues with type checking.
import asyncio  # noqa: E402
