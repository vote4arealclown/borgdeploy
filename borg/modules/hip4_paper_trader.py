"""Daily paper-trading logic for HIP-4 crypto binary options.

Each day Borg picks the strongest HIP-4 daily binary signal among BTC, ETH,
SOL, and HYPE and records a $1 paper trade.  Trades are settled when the
option expires by comparing the underlying spot price to the strike.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

import httpx

from borg.db import Database, db
from borg.modules.hip4 import fetch_all_mids

logger = logging.getLogger(__name__)

DEFAULT_STAKE = 1.0
CRYPTO_UNDERLYINGS = {"BTC", "ETH", "SOL", "HYPE"}


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    return value


def _normalise_prediction(p: dict[str, Any]) -> dict[str, Any]:
    """Add direction/confidence from implied probability if missing."""
    p = dict(p)
    prob = float(p.get("implied_probability", 0))
    if "direction" not in p or "confidence" not in p:
        if prob > 0.5:
            p["direction"] = "up"
            p["confidence"] = prob * 100.0
        elif prob < 0.5:
            p["direction"] = "down"
            p["confidence"] = (1.0 - prob) * 100.0
        else:
            p["direction"] = "flat"
            p["confidence"] = 50.0
    return p


def select_daily_trade(predictions: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Pick the single strongest crypto HIP-4 prediction for the day.

    Selection is simply highest confidence (market-implied probability of the
    predicted direction).  Returns None if no crypto predictions are supplied.
    """
    candidates = [
        _normalise_prediction(p)
        for p in predictions
        if p.get("underlying") in CRYPTO_UNDERLYINGS
        and _normalise_prediction(p).get("direction") in ("up", "down")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: float(p.get("confidence", 0)))


def build_paper_trade(
    prediction: dict[str, Any],
    stake: float = DEFAULT_STAKE,
) -> dict[str, Any]:
    """Convert a HIP-4 prediction into a paper-trade record.

    Buys $1 worth of YES if direction is up, or $1 worth of NO if direction is
    down.  Payout is 1 USDC per token if the option resolves in our favour.
    """
    prediction = _normalise_prediction(prediction)
    underlying = prediction["underlying"]
    direction = prediction["direction"]
    target_price = float(prediction.get("target_price", 0))
    yes_price = float(prediction.get("yes_price", 0))
    no_price = float(prediction.get("no_price", 0))
    expiry = prediction.get("expiry") or prediction.get("expiry_dt")

    if direction == "up":
        side = "YES"
        token_price = yes_price
    else:
        side = "NO"
        token_price = no_price

    quantity = stake / token_price if token_price > 0 else 0.0
    potential_payout = quantity  # each token pays 1 USDC if correct

    return {
        "trade_date": _to_date(expiry).isoformat(),
        "underlying": underlying,
        "direction": direction,
        "side": side,
        "target_price": target_price,
        "entry_price": float(prediction.get("target_price", 0)),  # strike
        "token_price": token_price,
        "quantity": quantity,
        "stake": stake,
        "potential_payout": potential_payout,
        "expiry": expiry.isoformat() if isinstance(expiry, (datetime, date)) else expiry,
    }


def _evaluate_outcome(trade: dict[str, Any], settle_price: float) -> tuple[str, float]:
    """Return (outcome, pnl) for a settled trade."""
    direction = trade["direction"]
    target = float(trade["target_price"])
    stake = float(trade["stake"])
    potential = float(trade["potential_payout"])

    if direction == "up":
        win = settle_price >= target
    else:
        win = settle_price < target

    pnl = potential - stake if win else -stake
    return ("win", pnl) if win else ("loss", pnl)


async def create_daily_paper_trade(
    predictions: list[dict[str, Any]],
    database: Optional[Database] = None,
    stake: float = DEFAULT_STAKE,
) -> Optional[dict[str, Any]]:
    """Create one paper trade for the strongest crypto option of the day.

    Skips if a trade already exists for the expiry date.
    """
    database = database or db
    prediction = select_daily_trade(predictions)
    if prediction is None:
        return None

    trade = build_paper_trade(prediction, stake=stake)
    if database.has_paper_trade_for_date(trade["trade_date"]):
        return None

    trade_id = database.insert_paper_trade(trade)
    if trade_id:
        trade["id"] = trade_id
        logger.info(
            "Opened HIP-4 paper trade #%s: %s %s %s @ strike %.4f, stake $%.2f",
            trade_id,
            trade["underlying"],
            trade["side"],
            trade["direction"].upper(),
            trade["target_price"],
            trade["stake"],
        )
        return trade
    return None


async def settle_open_paper_trades(
    database: Optional[Database] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> list[dict[str, Any]]:
    """Settle any paper trades whose expiry has passed.

    Uses Hyperliquid's all-mids snapshot for the underlying spot price at
    settlement time.  This is an approximation of the official oracle print.
    """
    database = database or db
    open_trades = database.get_open_paper_trades()
    if not open_trades:
        return []

    now = datetime.now(timezone.utc)
    ready = [t for t in open_trades if _to_datetime(t["expiry"]) <= now]
    if not ready:
        return []

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()
    settled: list[dict[str, Any]] = []
    try:
        mids = await fetch_all_mids(client)
        for trade in ready:
            underlying = trade["underlying"]
            settle_price = mids.get(underlying)
            if settle_price is None:
                logger.warning("No mid price for %s; skipping paper-trade settlement", underlying)
                continue
            outcome, pnl = _evaluate_outcome(trade, settle_price)
            database.settle_paper_trade(int(trade["id"]), settle_price, outcome, pnl)
            settled.append({
                "id": trade["id"],
                "underlying": underlying,
                "outcome": outcome,
                "pnl": pnl,
                "settle_price": settle_price,
            })
            logger.info(
                "Settled HIP-4 paper trade #%s: %s %s @ %.4f -> %s PnL %.4f",
                trade["id"],
                underlying,
                trade["side"],
                settle_price,
                outcome.upper(),
                pnl,
            )
    finally:
        if own_client:
            await client.aclose()
    return settled


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError(f"Cannot convert {value!r} to datetime")
