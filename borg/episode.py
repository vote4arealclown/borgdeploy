"""Episode capture: turn forecasts and market snapshots into structured episodes."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from borg.regime import detect_regime
from borg.schemas import Episode, EpisodeOutcome, EpisodeSignal


def _json_safe(obj: Any) -> Any:
    """Recursively make a value JSON-serializable."""
    from datetime import date, datetime

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _format_trigger(features: dict[str, Any]) -> str:
    """Build a concise trigger description from market features."""
    parts: list[str] = []
    rsi = features.get("rsi")
    if rsi is not None:
        parts.append(f"rsi={float(rsi):.1f}")
    volatility = features.get("volatility")
    if volatility is not None:
        parts.append(f"vol={float(volatility):.4f}")
    trend = features.get("trend")
    if trend is not None:
        parts.append(f"trend={trend}")
    return " ".join(parts) or "no_features"


def episode_from_forecast(
    forecast: dict[str, Any],
    market_data: dict[str, Any],
    actor: str = "unknown",
    outcome: Optional[dict[str, Any]] = None,
) -> Episode:
    """Build an Episode from a stored forecast and its market context.

    Parameters
    ----------
    forecast:
        Forecast row from the database (must contain ``direction``,
        ``confidence``, ``rationale``, ``created_at``).
    market_data:
        Market snapshot passed to the strategy, including ``candles`` and
        ``features``.
    actor:
        Strategy name that produced the forecast.
    outcome:
        Optional resolved outcome dict with ``win``, ``correct``,
        ``horizon_return``, ``outcome``.
    """
    candles = market_data.get("candles", [])
    features = market_data.get("features", {})
    regime = detect_regime(candles)

    created_at = forecast.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    timestamp = created_at or datetime.now(timezone.utc)

    signal = EpisodeSignal(
        direction=forecast.get("direction", "flat"),
        confidence=float(forecast.get("confidence", 0.0)),
        rationale=forecast.get("rationale"),
        size_multiplier=1.0,
    )

    market_state = _json_safe({
        "latest_close": float(candles[0]["close"]) if candles else None,
        "latest_high": float(candles[0]["high"]) if candles else None,
        "latest_low": float(candles[0]["low"]) if candles else None,
        "latest_volume": float(candles[0]["volume"]) if candles else None,
        "features": features,
    })

    episode_outcome: Optional[EpisodeOutcome] = None
    if outcome:
        episode_outcome = EpisodeOutcome(
            win=bool(outcome.get("win")),
            correct=bool(outcome.get("correct")),
            resolved_at=outcome.get("resolved_at"),
            horizon_return=outcome.get("horizon_return"),
            outcome=outcome.get("outcome"),
        )

    return Episode(
        timestamp=timestamp,
        actor=actor,
        trigger=_format_trigger(features),
        market_state=market_state,
        regime=regime,
        trade_signal=signal,
        executed=True,
        outcome=episode_outcome,
    )


def market_state_summary(candles: list[dict[str, Any]], features: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe market-state snapshot."""
    if not candles:
        return {"features": features}
    latest = candles[0]
    return {
        "latest_close": float(latest.get("close", 0.0)),
        "latest_high": float(latest.get("high", 0.0)),
        "latest_low": float(latest.get("low", 0.0)),
        "latest_volume": float(latest.get("volume", 0.0)),
        "candle_count": len(candles),
        "features": features,
    }
