"""Daily strategy signal and performance report."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from borg.config import settings
from borg.db import Database, db


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _today_range() -> tuple[str, str]:
    start = datetime(_today().year, _today().month, _today().day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def generate_strategy_report(database: Database = db) -> dict[str, Any]:
    """Aggregate today's strategy signals from stored forecasts.

    Returns a dict with per-strategy and per-symbol signal counts, plus a
    summary of the most recent signals.
    """
    start, end = _today_range()
    rows = database.fetchall(
        """
        SELECT symbol, direction, confidence, reasoning_output, created_at
        FROM forecasts
        WHERE created_at >= %s AND created_at < %s
        ORDER BY created_at DESC
        """
        if database.is_postgres
        else """
        SELECT symbol, direction, confidence, reasoning_output, created_at
        FROM forecasts
        WHERE created_at >= ? AND created_at < ?
        ORDER BY created_at DESC
        """,
        (start, end),
    )

    by_strategy: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"buy": 0, "sell": 0, "hold": 0, "signals": []}
    )
    by_symbol: dict[str, dict[str, int]] = defaultdict(lambda: {"buy": 0, "sell": 0, "hold": 0})

    for row in rows:
        symbol = row["symbol"]
        direction = row["direction"]
        reasoning = row.get("reasoning_output") or {}
        signals = reasoning.get("signals_input") or []
        created = row["created_at"]

        by_symbol[symbol][direction] = by_symbol[symbol].get(direction, 0) + 1

        for signal in signals:
            strategy = signal.get("strategy", "unknown")
            action = signal.get("action", "HOLD").lower()
            by_strategy[strategy][action] = by_strategy[strategy].get(action, 0) + 1
            by_strategy[strategy]["signals"].append(
                {
                    "symbol": symbol,
                    "action": action,
                    "confidence": signal.get("confidence", 0.0),
                    "time": str(created),
                }
            )

    return {
        "date": _today().isoformat(),
        "total_forecasts": len(rows),
        "by_symbol": dict(by_symbol),
        "by_strategy": {
            name: {
                "buy": counts["buy"],
                "sell": counts["sell"],
                "hold": counts["hold"],
                "latest_signals": counts["signals"][:5],
            }
            for name, counts in by_strategy.items()
        },
        "active_strategies": list(settings.symbol_list),
    }
