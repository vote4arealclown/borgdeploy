"""Push Borg data to Databricks SQL/Delta tables for external dashboards.

Requires environment variables (or config-supplied env names):
    DATABRICKS_HOST          e.g. https://<workspace>.cloud.databricks.com
    DATABRICKS_TOKEN         Databricks personal access token
    DATABRICKS_WAREHOUSE_ID  SQL warehouse ID

Tables are recreated/upserted idempotently on each export so the dashboard
always sees the latest snapshot.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from borg.config import settings
from borg.db import Database, db
from borg.events import EventLog, event_log

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def _client() -> Optional[WorkspaceClient]:
    """Build a Databricks SDK client from settings/env."""
    host = settings.databricks_host or _env("DATABRICKS_HOST")
    token = settings.databricks_token or _env("DATABRICKS_TOKEN")
    if not host or not token:
        return None
    return WorkspaceClient(host=host, token=token)


def _warehouse_id() -> Optional[str]:
    return settings.databricks_warehouse_id or _env("DATABRICKS_WAREHOUSE_ID")


def _table_name(key: str) -> str:
    return settings.databricks_tables.get(key, f"borg_{key}")


def _fully_qualified(table: str) -> str:
    return f"`{settings.databricks_catalog}`.`{settings.databricks_schema}`.`{table}`"


def _sql_literal(value: Any) -> str:
    """Format a Python value for a Databricks SQL statement."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, datetime):
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    if isinstance(value, date):
        return f"'{value.isoformat()}'"
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _rows_to_insert_sql(table: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Build a batched INSERT ... VALUES statement."""
    header = f"INSERT INTO {_fully_qualified(table)} ({', '.join(f'`{c}`' for c in columns)}) VALUES "
    values: list[str] = []
    for row in rows:
        vals = ", ".join(_sql_literal(row.get(c)) for c in columns)
        values.append(f"({vals})")
    return header + ", ".join(values)


def _execute(
    client: WorkspaceClient,
    warehouse_id: str,
    statement: str,
    timeout_seconds: int = 45,
) -> dict[str, Any]:
    # Databricks only allows wait timeouts between 5 s and 50 s.
    timeout_seconds = min(max(timeout_seconds, 5), 50)
    """Run a SQL statement on the Databricks warehouse and wait for it to finish."""
    resp = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog=settings.databricks_catalog,
        schema=settings.databricks_schema,
        statement=statement,
        wait_timeout=f"{timeout_seconds}s",
    )
    status = resp.status
    if status and status.state in (StatementState.FAILED, StatementState.CANCELED):
        raise RuntimeError(f"Databricks SQL failed: {status.error}")
    return {
        "state": status.state.value if status and status.state else "unknown",
        "statement": statement[:200],
    }


def _export_table(
    client: WorkspaceClient,
    warehouse_id: str,
    table_key: str,
    create_sql: str,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> dict[str, Any]:
    """Create a table and insert rows in batches."""
    table = _table_name(table_key)
    _execute(client, warehouse_id, create_sql)

    if not rows:
        return {"table": table, "inserted": 0, "batches": 0}

    # Clear existing rows so the published table is an idempotent snapshot.
    _execute(client, warehouse_id, f"DELETE FROM {_fully_qualified(table)}")

    inserted = 0
    batches = 0
    for i in range(0, len(rows), DEFAULT_BATCH_SIZE):
        batch = rows[i : i + DEFAULT_BATCH_SIZE]
        statement = _rows_to_insert_sql(table, columns, batch)
        _execute(client, warehouse_id, statement)
        inserted += len(batch)
        batches += 1

    return {"table": table, "inserted": inserted, "batches": batches}


def _fetch_forecasts(database: Database, limit: int = 10_000) -> list[dict[str, Any]]:
    rows = database.fetchall(
        """
        SELECT id, symbol, horizon_s, direction, confidence, rationale,
               features, outcome, correct, created_at, resolved_at
        FROM forecasts
        ORDER BY created_at DESC
        LIMIT %s
        """
        if database.is_postgres
        else """
        SELECT id, symbol, horizon_s, direction, confidence, rationale,
               features, outcome, correct, created_at, resolved_at
        FROM forecasts
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        r = database._row_to_dict(row)
        out.append({
            "id": r["id"],
            "symbol": r["symbol"],
            "horizon_s": r["horizon_s"],
            "direction": r["direction"],
            "confidence": float(r["confidence"]),
            "rationale": r.get("rationale"),
            "outcome": r.get("outcome"),
            "correct": bool(r.get("correct")) if r.get("correct") is not None else None,
            "created_at": r["created_at"],
            "resolved_at": r.get("resolved_at"),
        })
    return out


def _fetch_hip4(database: Database) -> list[dict[str, Any]]:
    rows = database.fetchall("SELECT * FROM hip4_predictions ORDER BY expiry DESC", ())
    out: list[dict[str, Any]] = []
    for r in (lambda rows_iter: [database._row_to_dict(r) for r in rows_iter])(rows):
        out.append({
            "id": r["id"],
            "underlying": r["underlying"],
            "outcome_id": r["outcome_id"],
            "expiry": r["expiry"],
            "target_price": float(r["target_price"]),
            "yes_price": float(r["yes_price"]),
            "no_price": float(r["no_price"]),
            "implied_probability": float(r["implied_probability"]),
            "direction": r["direction"],
            "confidence": float(r["confidence"]),
            "rationale": r.get("rationale"),
            "exported_at": datetime.now(timezone.utc),
        })
    return out


def _fetch_paper_trades(database: Database) -> list[dict[str, Any]]:
    rows = database.fetchall("SELECT * FROM paper_trades ORDER BY trade_date DESC", ())
    out: list[dict[str, Any]] = []
    for r in (lambda rows_iter: [database._row_to_dict(r) for r in rows_iter])(rows):
        out.append({
            "id": r["id"],
            "trade_date": r["trade_date"],
            "underlying": r["underlying"],
            "direction": r["direction"],
            "side": r["side"],
            "target_price": float(r["target_price"]),
            "entry_price": float(r["entry_price"]),
            "token_price": float(r["token_price"]),
            "quantity": float(r["quantity"]),
            "stake": float(r["stake"]),
            "potential_payout": float(r["potential_payout"]),
            "expiry": r["expiry"],
            "outcome": r.get("outcome"),
            "settle_price": float(r["settle_price"]) if r.get("settle_price") is not None else None,
            "pnl": float(r["pnl"]) if r.get("pnl") is not None else None,
            "settled_at": r.get("settled_at"),
            "created_at": r["created_at"],
        })
    return out


def _fetch_candles(database: Database, limit_per_symbol: int = 200) -> list[dict[str, Any]]:
    """Fetch the most recent candles per symbol to keep the published set small."""
    symbols = database.fetchall("SELECT DISTINCT symbol FROM market_candles", ())
    out: list[dict[str, Any]] = []
    for row in (lambda rows_iter: [database._row_to_dict(r) for r in rows_iter])(symbols):
        sym = row["symbol"]
        rows = database.fetchall(
            "SELECT id, symbol, ts, open, high, low, close, volume, created_at FROM market_candles WHERE symbol = %s ORDER BY ts DESC LIMIT %s"
            if database.is_postgres
            else "SELECT id, symbol, ts, open, high, low, close, volume, created_at FROM market_candles WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
            (sym, limit_per_symbol),
        )
        for r in (lambda rows_iter: [database._row_to_dict(r) for r in rows_iter])(rows):
            out.append({
                "id": r["id"],
                "symbol": r["symbol"],
                "ts": r["ts"],
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
                "created_at": r["created_at"],
            })
    return out


def _fetch_reports(database: Database, limit: int = 1_000) -> list[dict[str, Any]]:
    rows = database.fetchall(
        """
        SELECT id, slug, title, category, report_date, description, content_json, status, created_at, updated_at
        FROM reports
        ORDER BY report_date DESC, created_at DESC
        LIMIT %s
        """
        if database.is_postgres
        else """
        SELECT id, slug, title, category, report_date, description, content_json, status, created_at, updated_at
        FROM reports
        ORDER BY report_date DESC, created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        r = database._row_to_dict(row)
        out.append({
            "id": r["id"],
            "slug": r["slug"],
            "title": r["title"],
            "category": r["category"],
            "report_date": r["report_date"],
            "description": r.get("description"),
            "content_json": json.dumps(database._from_json(r.get("content_json")) or {}),
            "status": r.get("status"),
            "created_at": r["created_at"],
            "updated_at": r.get("updated_at"),
            "exported_at": datetime.now(timezone.utc),
        })
    return out


async def export_all(
    database: Optional[Database] = None,
    events: Optional[EventLog] = None,
) -> dict[str, Any]:
    """Export reports, forecasts, HIP-4 predictions, paper trades, and recent candles to Databricks."""
    database = database or db
    events = events or event_log

    if not settings.databricks_enabled:
        return {"status": "disabled", "message": "Databricks export is disabled in config"}

    client = _client()
    if client is None:
        msg = "Databricks host/token not configured; set DATABRICKS_HOST and DATABRICKS_TOKEN"
        logger.warning(msg)
        return {"status": "error", "message": msg}

    warehouse_id = _warehouse_id()
    if not warehouse_id:
        msg = "DATABRICKS_WAREHOUSE_ID not configured"
        logger.warning(msg)
        return {"status": "error", "message": msg}

    results: dict[str, Any] = {}
    try:
        # Reports
        reports = _fetch_reports(database)
        results["reports"] = _export_table(
            client,
            warehouse_id,
            "reports",
            f"""
            CREATE TABLE IF NOT EXISTS {_fully_qualified(_table_name('reports'))} (
                id BIGINT,
                slug STRING,
                title STRING,
                category STRING,
                report_date DATE,
                description STRING,
                content_json STRING,
                status STRING,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                exported_at TIMESTAMP
            ) USING DELTA
            """,
            reports,
            ["id", "slug", "title", "category", "report_date", "description", "content_json", "status", "created_at", "updated_at", "exported_at"],
        )

        # Forecasts
        forecasts = _fetch_forecasts(database)
        results["forecasts"] = _export_table(
            client,
            warehouse_id,
            "forecasts",
            f"""
            CREATE TABLE IF NOT EXISTS {_fully_qualified(_table_name('forecasts'))} (
                id BIGINT,
                symbol STRING,
                horizon_s BIGINT,
                direction STRING,
                confidence DOUBLE,
                rationale STRING,
                outcome STRING,
                correct BOOLEAN,
                created_at TIMESTAMP,
                resolved_at TIMESTAMP
            ) USING DELTA
            """,
            forecasts,
            ["id", "symbol", "horizon_s", "direction", "confidence", "rationale", "outcome", "correct", "created_at", "resolved_at"],
        )

        # HIP-4 predictions
        hip4 = _fetch_hip4(database)
        results["hip4_predictions"] = _export_table(
            client,
            warehouse_id,
            "hip4_predictions",
            f"""
            CREATE TABLE IF NOT EXISTS {_fully_qualified(_table_name('hip4_predictions'))} (
                id BIGINT,
                underlying STRING,
                outcome_id BIGINT,
                expiry TIMESTAMP,
                target_price DOUBLE,
                yes_price DOUBLE,
                no_price DOUBLE,
                implied_probability DOUBLE,
                direction STRING,
                confidence DOUBLE,
                rationale STRING,
                exported_at TIMESTAMP
            ) USING DELTA
            """,
            hip4,
            ["id", "underlying", "outcome_id", "expiry", "target_price", "yes_price", "no_price", "implied_probability", "direction", "confidence", "rationale", "exported_at"],
        )

        # Paper trades
        paper = _fetch_paper_trades(database)
        results["paper_trades"] = _export_table(
            client,
            warehouse_id,
            "paper_trades",
            f"""
            CREATE TABLE IF NOT EXISTS {_fully_qualified(_table_name('paper_trades'))} (
                id BIGINT,
                trade_date DATE,
                underlying STRING,
                direction STRING,
                side STRING,
                target_price DOUBLE,
                entry_price DOUBLE,
                token_price DOUBLE,
                quantity DOUBLE,
                stake DOUBLE,
                potential_payout DOUBLE,
                expiry TIMESTAMP,
                outcome STRING,
                settle_price DOUBLE,
                pnl DOUBLE,
                settled_at TIMESTAMP,
                created_at TIMESTAMP
            ) USING DELTA
            """,
            paper,
            ["id", "trade_date", "underlying", "direction", "side", "target_price", "entry_price", "token_price", "quantity", "stake", "potential_payout", "expiry", "outcome", "settle_price", "pnl", "settled_at", "created_at"],
        )

        # Candles
        candles = _fetch_candles(database)
        results["candles"] = _export_table(
            client,
            warehouse_id,
            "candles",
            f"""
            CREATE TABLE IF NOT EXISTS {_fully_qualified(_table_name('candles'))} (
                id BIGINT,
                symbol STRING,
                ts TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                created_at TIMESTAMP
            ) USING DELTA
            """,
            candles,
            ["id", "symbol", "ts", "open", "high", "low", "close", "volume", "created_at"],
        )

        summary = {k: f"{v['inserted']} rows" for k, v in results.items()}
        events.emit(
            f"Databricks export complete: {summary}",
            category="databricks",
            phase="act",
            metadata=results,
        )
        return {"status": "ok", "results": results}
    except Exception as exc:
        logger.exception("Databricks export failed")
        events.emit(f"Databricks export failed: {exc}", category="databricks", phase="act", level="ERROR")
        return {"status": "error", "message": str(exc)}
