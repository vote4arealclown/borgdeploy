"""Database abstraction: PostgreSQL + pgvector primary, SQLite fallback."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from borg.config import settings

# Optional PostgreSQL imports; gracefully degrade to SQLite if missing.
try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import JsonbDumper

    # Register automatic dict → JSONB adaptation (lists stay as Postgres arrays)
    psycopg.adapters.register_dumper(dict, JsonbDumper)

    POSTGRES_AVAILABLE = True
except Exception:
    POSTGRES_AVAILABLE = False


def _dedent_sql(sql: str) -> str:
    lines = sql.splitlines()
    stripped = [line for line in lines if line.strip()]
    return "\n".join(stripped)


class Database:
    """Unified DB interface with PostgreSQL primary and SQLite fallback."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        # Each Database instance gets its own thread-local connection cache so
        # multiple instances (e.g. in tests) do not share connections.
        self._local = threading.local()
        raw_url = database_url or settings.database_url
        # Normalize SQLAlchemy-style URLs for psycopg
        if raw_url.startswith("postgresql+"):
            raw_url = raw_url.replace("postgresql+psycopg", "postgresql", 1)
        self.database_url = raw_url
        self.is_postgres = self.database_url.startswith("postgresql")
        if self.is_postgres and not POSTGRES_AVAILABLE:
            raise RuntimeError("PostgreSQL requested but psycopg is not installed")
        if not self.is_postgres:
            self.sqlite_path = Path(self.database_url.replace("sqlite:///", ""))
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def placeholder(self) -> str:
        """Return the parameter placeholder for the active database backend."""
        return "%s" if self.is_postgres else "?"

    def close(self) -> None:
        """Close any open thread-local connections for this instance."""
        if self.is_postgres:
            conn = getattr(self._local, "pg_conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.pg_conn = None
        else:
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.conn = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # Connection management --------------------------------------------------

    def _connection(self) -> Any:
        if self.is_postgres:
            # psycopg ConnectionPool is preferred; for prototype use per-thread conn.
            conn = getattr(self._local, "pg_conn", None)
            if conn is None or conn.closed:
                conn = psycopg.connect(self.database_url, row_factory=dict_row)
                self._local.pg_conn = conn
            return conn

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def _cursor(self):
        conn = self._connection()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[Any]:
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        with self._cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # Schema -----------------------------------------------------------------

    def _init_schema(self) -> None:
        if self.is_postgres:
            self._init_postgres()
        else:
            self._init_sqlite()

    def _init_postgres(self) -> None:
        schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
        conn = self._connection()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute("ALTER TABLE versions ADD COLUMN IF NOT EXISTS new_content TEXT")
        conn.autocommit = False
        conn.commit()

    def _init_sqlite(self) -> None:
        # SQLite fallback schema without vector/pg-specific types.
        schema = """
        CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
        CREATE TABLE IF NOT EXISTS goals (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, description TEXT, priority INTEGER, status TEXT, created_at TEXT, updated_at TEXT);
        CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, goal_id INTEGER, kind TEXT, payload TEXT, status TEXT, result TEXT, created_by TEXT, created_at TEXT, started_at TEXT, finished_at TEXT);
        CREATE TABLE IF NOT EXISTS market_candles (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, created_at TEXT, UNIQUE(symbol, ts));
        CREATE TABLE IF NOT EXISTS forecasts (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, symbol TEXT, horizon_s INTEGER, direction TEXT, confidence REAL, rationale TEXT, features TEXT, outcome TEXT, correct INTEGER, reasoning_output TEXT, created_at TEXT, resolved_at TEXT);
        CREATE TABLE IF NOT EXISTS reflections (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, critique TEXT, score REAL, created_at TEXT);
        CREATE TABLE IF NOT EXISTS learnings (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER, summary TEXT, detail TEXT, embedding TEXT, tags TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, content TEXT, embedding TEXT, source TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            actor TEXT NOT NULL,
            trigger TEXT,
            market_state TEXT,
            regime TEXT NOT NULL,
            trade_signal TEXT,
            executed INTEGER DEFAULT 1,
            outcome TEXT,
            reasoning_output TEXT,
            embedding TEXT,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_episodes_actor_regime ON episodes(actor, regime, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
        CREATE TABLE IF NOT EXISTS reasoning_audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_id INTEGER,
            reasoning_decision TEXT,
            reasoning_confidence REAL,
            reasoning_why TEXT,
            outcome_win INTEGER,
            outcome_pnl REAL,
            calibration_error REAL,
            metadata TEXT,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_reasoning_audits_created ON reasoning_audits(created_at DESC);
        CREATE TABLE IF NOT EXISTS conscious_summaries (id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT, context TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS consciousness_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period TEXT NOT NULL,
            report_date TEXT NOT NULL,
            report_text TEXT NOT NULL,
            score REAL,
            metadata TEXT,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_consciousness_reports_period_date ON consciousness_reports(period, report_date DESC);
        CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, category TEXT, phase TEXT, message TEXT, symbol TEXT, metadata TEXT);
        CREATE TABLE IF NOT EXISTS hip4_predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, underlying TEXT, outcome_id INTEGER, expiry TEXT, target_price REAL, yes_price REAL, no_price REAL, implied_probability REAL, direction TEXT, confidence REAL, rationale TEXT, UNIQUE(underlying, expiry));
        CREATE TABLE IF NOT EXISTS paper_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date TEXT, underlying TEXT, direction TEXT, side TEXT, target_price REAL, entry_price REAL, token_price REAL, quantity REAL, stake REAL, potential_payout REAL, expiry TEXT, outcome TEXT DEFAULT 'pending', settle_price REAL, pnl REAL, settled_at TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(trade_date));
        CREATE TABLE IF NOT EXISTS versions (id INTEGER PRIMARY KEY AUTOINCREMENT, module TEXT, version TEXT, path TEXT, diff TEXT, new_content TEXT, status TEXT, created_at TEXT, applied_at TEXT);
        CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT, action TEXT, detail TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, message TEXT, ts TEXT, metadata TEXT);
        CREATE TABLE IF NOT EXISTS brain_cycles (id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT, finished_at TEXT, status TEXT, summary TEXT);
        CREATE TABLE IF NOT EXISTS distributed_locks (lock_name TEXT PRIMARY KEY, holder TEXT NOT NULL, expires_at TEXT NOT NULL, acquired_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT UNIQUE, title TEXT, category TEXT, report_date TEXT, description TEXT, content_json TEXT, status TEXT, created_at TEXT, updated_at TEXT);
        CREATE TABLE IF NOT EXISTS market_deltas (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, price REAL, change_pct REAL, category TEXT, report_date TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS scheduled_events (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, category TEXT, impact TEXT, event_time TEXT, description TEXT, tags TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS inventory_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            name TEXT,
            entry_type TEXT NOT NULL,
            size_bytes INTEGER,
            content_hash TEXT,
            content TEXT,
            metadata TEXT,
            assimilation_score REAL,
            assimilation_status TEXT NOT NULL DEFAULT 'pending',
            assimilation_reason TEXT,
            version_id INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_inventory_source ON inventory_entries(source);
        CREATE INDEX IF NOT EXISTS idx_inventory_status ON inventory_entries(assimilation_status);
        CREATE INDEX IF NOT EXISTS idx_inventory_score ON inventory_entries(assimilation_score DESC);
        """
        conn = self._connection()
        conn.executescript(schema)
        try:
            conn.execute("ALTER TABLE versions ADD COLUMN new_content TEXT")
        except Exception:
            pass  # column already exists
        conn.execute(
            "INSERT OR IGNORE INTO kv (key, value, updated_at) VALUES (?, ?, ?)",
            (
                "mission",
                json.dumps({"text": settings.mission}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    # Serializers ------------------------------------------------------------

    def _to_json(self, obj: Any) -> Optional[Any]:
        if obj is None:
            return None
        if self.is_postgres:
            return obj
        return json.dumps(obj)

    def _from_json(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _insert(self, sql: str, params: tuple[Any, ...]) -> int:
        """Insert and return generated id. Uses RETURNING on Postgres, lastrowid on SQLite."""
        if self.is_postgres:
            row = self.fetchone(sql + " RETURNING id", params)
            return row["id"] if row else 0
        cur = self.execute(sql, params)
        return cur.lastrowid or 0

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if self.is_postgres:
            return dict(row)
        return dict(row)

    # KV ----------------------------------------------------------------------

    def get_kv(self, key: str) -> Optional[Any]:
        row = self.fetchone("SELECT value FROM kv WHERE key = %s" if self.is_postgres else "SELECT value FROM kv WHERE key = ?", (key,))
        if row:
            return self._from_json(row["value"])
        return None

    def set_kv(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv (key, value, updated_at) VALUES (%s, %s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at"
            if self.is_postgres
            else "INSERT OR REPLACE INTO kv (key, value, updated_at) VALUES (?, ?, ?)",
            (key, self._to_json(value), self._now()),
        )

    # Goals ------------------------------------------------------------------

    def create_goal(self, title: str, description: str = "", priority: int = 50) -> int:
        return self._insert(
            "INSERT INTO goals (title, description, priority, status, created_at, updated_at) VALUES (%s, %s, %s, 'active', %s, %s)"
            if self.is_postgres
            else "INSERT INTO goals (title, description, priority, status, created_at, updated_at) VALUES (?, ?, ?, 'active', ?, ?)",
            (title, description, priority, self._now(), self._now()),
        )

    def list_goals(self, status: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
        if status:
            rows = self.fetchall("SELECT * FROM goals WHERE status = %s ORDER BY priority DESC, created_at DESC LIMIT %s" if self.is_postgres else "SELECT * FROM goals WHERE status = ? ORDER BY priority DESC, created_at DESC LIMIT ?", (status, limit))
        else:
            rows = self.fetchall("SELECT * FROM goals ORDER BY priority DESC, created_at DESC LIMIT %s" if self.is_postgres else "SELECT * FROM goals ORDER BY priority DESC, created_at DESC LIMIT ?", (limit,))
        return [self._row_to_dict(r) for r in rows]

    def update_goal_status(self, goal_id: int, status: str) -> None:
        self.execute(
            "UPDATE goals SET status = %s, updated_at = %s WHERE id = %s"
            if self.is_postgres
            else "UPDATE goals SET status = ?, updated_at = ? WHERE id = ?",
            (status, self._now(), goal_id),
        )

    # Tasks ------------------------------------------------------------------

    def create_task(
        self,
        kind: str,
        payload: dict[str, Any],
        goal_id: Optional[int] = None,
        status: str = "pending",
        created_by: str = "system",
    ) -> int:
        return self._insert(
            "INSERT INTO tasks (goal_id, kind, payload, status, created_by, created_at) VALUES (%s, %s, %s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO tasks (goal_id, kind, payload, status, created_by, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (goal_id, kind, self._to_json(payload), status, created_by, self._now()),
        )

    def get_task(self, task_id: int) -> Optional[dict[str, Any]]:
        row = self.fetchone("SELECT * FROM tasks WHERE id = %s" if self.is_postgres else "SELECT * FROM tasks WHERE id = ?", (task_id,))
        if row:
            d = self._row_to_dict(row)
            d["payload"] = self._from_json(d.get("payload"))
            d["result"] = self._from_json(d.get("result"))
            return d
        return None

    def list_tasks(
        self,
        status: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND status = %s" if self.is_postgres else " AND status = ?"
            params.append(status)
        if kind:
            sql += " AND kind = %s" if self.is_postgres else " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT %s" if self.is_postgres else " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.fetchall(sql, tuple(params))
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["payload"] = self._from_json(d.get("payload"))
            d["result"] = self._from_json(d.get("result"))
            out.append(d)
        return out

    def update_task_status(self, task_id: int, status: str, result: Optional[dict[str, Any]] = None) -> None:
        now = self._now()
        if status == "running":
            self.execute(
                "UPDATE tasks SET status = %s, started_at = %s WHERE id = %s"
                if self.is_postgres
                else "UPDATE tasks SET status = ?, started_at = ? WHERE id = ?",
                (status, now, task_id),
            )
        elif status in ("done", "failed", "rejected"):
            self.execute(
                "UPDATE tasks SET status = %s, finished_at = %s, result = %s WHERE id = %s"
                if self.is_postgres
                else "UPDATE tasks SET status = ?, finished_at = ?, result = ? WHERE id = ?",
                (status, now, self._to_json(result), task_id),
            )
        else:
            self.execute(
                "UPDATE tasks SET status = %s WHERE id = %s"
                if self.is_postgres
                else "UPDATE tasks SET status = ? WHERE id = ?",
                (status, task_id),
            )

    # Candles ----------------------------------------------------------------

    def insert_candle(self, candle: dict[str, Any]) -> int:
        if self.is_postgres:
            row = self.fetchone(
                "INSERT INTO market_candles (symbol, ts, open, high, low, close, volume, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (symbol, ts) DO NOTHING RETURNING id",
                (
                    candle["symbol"],
                    candle["ts"].isoformat() if isinstance(candle["ts"], datetime) else candle["ts"],
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"],
                    self._now(),
                ),
            )
            return row["id"] if row else 0
        cur = self.execute(
            "INSERT OR IGNORE INTO market_candles (symbol, ts, open, high, low, close, volume, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                candle["symbol"],
                candle["ts"].isoformat() if isinstance(candle["ts"], datetime) else candle["ts"],
                candle["open"],
                candle["high"],
                candle["low"],
                candle["close"],
                candle["volume"],
                self._now(),
            ),
        )
        return cur.lastrowid or 0

    def latest_candles(self, symbol: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT * FROM market_candles WHERE symbol = %s ORDER BY ts DESC LIMIT %s"
            if self.is_postgres
            else "SELECT * FROM market_candles WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
            (symbol, limit),
        )
        return [self._row_to_dict(r) for r in rows]

    def candles_before(self, symbol: str, before: datetime, limit: int = 50) -> list[dict[str, Any]]:
        """Return candles for a symbol up to (and including) a timestamp, newest first."""
        iso = before.isoformat() if isinstance(before, datetime) else before
        rows = self.fetchall(
            "SELECT * FROM market_candles WHERE symbol = %s AND ts <= %s ORDER BY ts DESC LIMIT %s"
            if self.is_postgres
            else "SELECT * FROM market_candles WHERE symbol = ? AND ts <= ? ORDER BY ts DESC LIMIT ?",
            (symbol, iso, limit),
        )
        return [self._row_to_dict(r) for r in rows]

    # Forecasts --------------------------------------------------------------

    def insert_forecast(self, forecast: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO forecasts (task_id, symbol, horizon_s, direction, confidence, rationale, features, reasoning_output, outcome, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)"
            if self.is_postgres
            else "INSERT INTO forecasts (task_id, symbol, horizon_s, direction, confidence, rationale, features, reasoning_output, outcome, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (
                forecast.get("task_id"),
                forecast["symbol"],
                forecast.get("horizon_s", 300),
                forecast["direction"],
                forecast["confidence"],
                forecast.get("rationale"),
                self._to_json(forecast.get("features")),
                self._to_json(forecast.get("reasoning_output")),
                self._now(),
            ),
        )

    def insert_hip4_prediction(self, prediction: dict[str, Any]) -> int:
        """Insert or update a HIP-4 binary-option prediction for an underlying/expiry pair."""
        if self.is_postgres:
            sql = """
                INSERT INTO hip4_predictions
                    (underlying, outcome_id, expiry, target_price, yes_price, no_price, implied_probability, direction, confidence, rationale)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (underlying, expiry) DO UPDATE SET
                    outcome_id = EXCLUDED.outcome_id,
                    target_price = EXCLUDED.target_price,
                    yes_price = EXCLUDED.yes_price,
                    no_price = EXCLUDED.no_price,
                    implied_probability = EXCLUDED.implied_probability,
                    direction = EXCLUDED.direction,
                    confidence = EXCLUDED.confidence,
                    rationale = EXCLUDED.rationale
                RETURNING id
            """
        else:
            sql = """
                INSERT OR REPLACE INTO hip4_predictions
                    (underlying, outcome_id, expiry, target_price, yes_price, no_price, implied_probability, direction, confidence, rationale)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        params = (
            prediction["underlying"],
            prediction["outcome_id"],
            prediction["expiry"].isoformat() if isinstance(prediction["expiry"], datetime) else prediction["expiry"],
            prediction["target_price"],
            prediction["yes_price"],
            prediction["no_price"],
            prediction["implied_probability"],
            prediction["direction"],
            prediction["confidence"],
            prediction.get("rationale"),
        )
        if self.is_postgres:
            row = self.fetchone(sql, params)
            return row["id"] if row else 0
        cur = self.execute(sql, params)
        return cur.lastrowid or 0

    def recent_hip4_predictions(self, underlying: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if underlying:
            rows = self.fetchall(
                "SELECT * FROM hip4_predictions WHERE underlying = %s ORDER BY expiry DESC LIMIT %s"
                if self.is_postgres
                else "SELECT * FROM hip4_predictions WHERE underlying = ? ORDER BY expiry DESC LIMIT ?",
                (underlying, limit),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM hip4_predictions ORDER BY expiry DESC LIMIT %s"
                if self.is_postgres
                else "SELECT * FROM hip4_predictions ORDER BY expiry DESC LIMIT ?",
                (limit,),
            )
        return [self._row_to_dict(r) for r in rows]

    def insert_paper_trade(self, trade: dict[str, Any]) -> int:
        """Record a daily HIP-4 paper trade."""
        if self.is_postgres:
            sql = """
                INSERT INTO paper_trades
                    (trade_date, underlying, direction, side, target_price, entry_price, token_price, quantity, stake, potential_payout, expiry, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (trade_date) DO NOTHING
                RETURNING id
            """
        else:
            sql = """
                INSERT OR IGNORE INTO paper_trades
                    (trade_date, underlying, direction, side, target_price, entry_price, token_price, quantity, stake, potential_payout, expiry, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        params = (
            trade["trade_date"],
            trade["underlying"],
            trade["direction"],
            trade["side"],
            trade["target_price"],
            trade["entry_price"],
            trade["token_price"],
            trade["quantity"],
            trade["stake"],
            trade["potential_payout"],
            trade["expiry"].isoformat() if isinstance(trade["expiry"], datetime) else trade["expiry"],
        )
        if not self.is_postgres:
            params += (self._now(),)
        if self.is_postgres:
            row = self.fetchone(sql, params)
            return row["id"] if row else 0
        cur = self.execute(sql, params)
        return cur.lastrowid or 0

    def get_open_paper_trades(self) -> list[dict[str, Any]]:
        """Return paper trades that have not been settled yet."""
        rows = self.fetchall(
            "SELECT * FROM paper_trades WHERE outcome = 'pending' ORDER BY expiry ASC"
            if self.is_postgres
            else "SELECT * FROM paper_trades WHERE outcome = 'pending' ORDER BY expiry ASC",
            (),
        )
        return [self._row_to_dict(r) for r in rows]

    def has_paper_trade_for_date(self, trade_date: str) -> bool:
        """Check whether a paper trade already exists for the given expiry date."""
        row = self.fetchone(
            "SELECT 1 FROM paper_trades WHERE trade_date = %s"
            if self.is_postgres
            else "SELECT 1 FROM paper_trades WHERE trade_date = ?",
            (trade_date,),
        )
        return row is not None

    def settle_paper_trade(self, trade_id: int, settle_price: float, outcome: str, pnl: float) -> None:
        """Mark a paper trade as settled with final price and PnL."""
        self.execute(
            "UPDATE paper_trades SET outcome = %s, settle_price = %s, pnl = %s, settled_at = NOW() WHERE id = %s"
            if self.is_postgres
            else "UPDATE paper_trades SET outcome = ?, settle_price = ?, pnl = ?, settled_at = ? WHERE id = ?",
            (outcome, settle_price, pnl, self._now(), trade_id) if not self.is_postgres else (outcome, settle_price, pnl, trade_id),
        )

    def recent_paper_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT * FROM paper_trades ORDER BY trade_date DESC LIMIT %s"
            if self.is_postgres
            else "SELECT * FROM paper_trades ORDER BY trade_date DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(r) for r in rows]

    def recent_forecasts(self, symbol: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if symbol:
            rows = self.fetchall(
                "SELECT * FROM forecasts WHERE symbol = %s ORDER BY created_at DESC LIMIT %s"
                if self.is_postgres
                else "SELECT * FROM forecasts WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM forecasts ORDER BY created_at DESC LIMIT %s"
                if self.is_postgres
                else "SELECT * FROM forecasts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["features"] = self._from_json(d.get("features"))
            d["reasoning_output"] = self._from_json(d.get("reasoning_output"))
            out.append(d)
        return out

    def unresolved_forecasts(self, before: Optional[datetime] = None) -> list[dict[str, Any]]:
        if before is None:
            before = datetime.now(timezone.utc)
        rows = self.fetchall(
            "SELECT * FROM forecasts WHERE outcome = 'pending' AND created_at < %s"
            if self.is_postgres
            else "SELECT * FROM forecasts WHERE outcome = 'pending' AND created_at < ?",
            (before.isoformat(),),
        )
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["features"] = self._from_json(d.get("features"))
            d["reasoning_output"] = self._from_json(d.get("reasoning_output"))
            out.append(d)
        return out

    def resolve_forecast(self, forecast_id: int, outcome: str, correct: bool) -> None:
        self.execute(
            "UPDATE forecasts SET outcome = %s, correct = %s, resolved_at = %s WHERE id = %s"
            if self.is_postgres
            else "UPDATE forecasts SET outcome = ?, correct = ?, resolved_at = ? WHERE id = ?",
            (outcome, correct, self._now(), forecast_id),
        )

    def update_forecast_reasoning(self, forecast_id: int, reasoning_output: dict[str, Any]) -> None:
        self.execute(
            "UPDATE forecasts SET reasoning_output = %s WHERE id = %s"
            if self.is_postgres
            else "UPDATE forecasts SET reasoning_output = ? WHERE id = ?",
            (self._to_json(reasoning_output), forecast_id),
        )

    def insert_reasoning_audit(self, audit: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO reasoning_audits (forecast_id, reasoning_decision, reasoning_confidence, reasoning_why, outcome_win, outcome_pnl, calibration_error, metadata, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO reasoning_audits (forecast_id, reasoning_decision, reasoning_confidence, reasoning_why, outcome_win, outcome_pnl, calibration_error, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                audit.get("forecast_id"),
                audit.get("reasoning_decision"),
                audit.get("reasoning_confidence"),
                audit.get("reasoning_why"),
                bool(audit.get("outcome_win")),
                audit.get("outcome_pnl"),
                audit.get("calibration_error"),
                self._to_json(audit.get("metadata")),
                self._now(),
            ),
        )

    def query_reasoning_audits(
        self,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM reasoning_audits WHERE 1=1"
        params: list[Any] = []
        if since:
            sql += " AND created_at >= %s" if self.is_postgres else " AND created_at >= ?"
            params.append(since)
        sql += " ORDER BY created_at DESC LIMIT %s" if self.is_postgres else " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.fetchall(sql, tuple(params))
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["metadata"] = self._from_json(d.get("metadata"))
            d["outcome_win"] = bool(d.get("outcome_win"))
            out.append(d)
        return out

    # Reflections ------------------------------------------------------------

    def insert_reflection(self, reflection: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO reflections (task_id, critique, score, created_at) VALUES (%s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO reflections (task_id, critique, score, created_at) VALUES (?, ?, ?, ?)",
            (reflection.get("task_id"), reflection.get("critique"), reflection.get("score"), self._now()),
        )

    # Learnings --------------------------------------------------------------

    def insert_learning(self, learning: dict[str, Any]) -> int:
        embedding = learning.get("embedding")
        if embedding is not None:
            embedding = json.dumps(embedding)
        tags = learning.get("tags")
        if self.is_postgres:
            return self._insert(
                "INSERT INTO learnings (task_id, summary, detail, embedding, tags, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    learning.get("task_id"),
                    learning["summary"],
                    learning.get("detail"),
                    embedding,
                    tags,
                    self._now(),
                ),
            )
        return self._insert(
            "INSERT INTO learnings (task_id, summary, detail, embedding, tags, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (learning.get("task_id"), learning["summary"], learning.get("detail"), embedding, json.dumps(tags) if tags else None, self._now()),
        )

    def recent_learnings(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT * FROM learnings ORDER BY created_at DESC LIMIT %s"
            if self.is_postgres
            else "SELECT * FROM learnings ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["embedding"] = self._from_json(d.get("embedding"))
            out.append(d)
        return out

    def search_learnings(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if self.is_postgres:
            rows = self.fetchall(
                "SELECT *, embedding <=> %s::vector AS distance FROM learnings ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_embedding, query_embedding, top_k),
            )
            return [self._row_to_dict(r) for r in rows]
        # SQLite fallback: brute-force cosine similarity
        rows = self.fetchall("SELECT * FROM learnings")
        scored = []
        for r in rows:
            emb = self._from_json(r["embedding"])
            if emb:
                scored.append((self._cosine_similarity(query_embedding, emb), self._row_to_dict(r)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # Memory -----------------------------------------------------------------

    def insert_memory(self, memory: dict[str, Any]) -> int:
        embedding = memory.get("embedding")
        if embedding is not None:
            embedding = json.dumps(embedding)
        if self.is_postgres:
            return self._insert(
                "INSERT INTO memory (kind, content, embedding, source, created_at) VALUES (%s, %s, %s, %s, %s)",
                (memory["kind"], memory["content"], embedding, memory.get("source"), self._now()),
            )
        return self._insert(
            "INSERT INTO memory (kind, content, embedding, source, created_at) VALUES (?, ?, ?, ?, ?)",
            (memory["kind"], memory["content"], embedding, memory.get("source"), self._now()),
        )

    def search_memory(self, query_embedding: list[float], kind: Optional[str] = None, top_k: int = 5) -> list[dict[str, Any]]:
        if self.is_postgres:
            if kind:
                rows = self.fetchall(
                    "SELECT *, embedding <=> %s::vector AS distance FROM memory WHERE kind = %s ORDER BY embedding <=> %s::vector LIMIT %s",
                    (query_embedding, kind, query_embedding, top_k),
                )
            else:
                rows = self.fetchall(
                    "SELECT *, embedding <=> %s::vector AS distance FROM memory ORDER BY embedding <=> %s::vector LIMIT %s",
                    (query_embedding, query_embedding, top_k),
                )
            return [self._row_to_dict(r) for r in rows]
        rows = self.fetchall("SELECT * FROM memory" + (" WHERE kind = ?" if kind else ""), (kind,) if kind else ())
        scored = []
        for r in rows:
            emb = self._from_json(r["embedding"])
            if emb:
                scored.append((self._cosine_similarity(query_embedding, emb), self._row_to_dict(r)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    # Episodes ---------------------------------------------------------------

    def insert_episode(self, episode: dict[str, Any]) -> int:
        embedding = episode.get("embedding")
        if embedding is not None:
            embedding = json.dumps(embedding)
        params = (
            episode["timestamp"].isoformat() if isinstance(episode["timestamp"], datetime) else episode["timestamp"],
            episode["actor"],
            episode.get("trigger"),
            self._to_json(episode.get("market_state")),
            episode["regime"],
            self._to_json(episode.get("trade_signal")),
            bool(episode.get("executed", True)),
            self._to_json(episode.get("outcome")),
            self._to_json(episode.get("reasoning_output")),
            embedding,
            self._now(),
        )
        if self.is_postgres:
            return self._insert(
                "INSERT INTO episodes (timestamp, actor, trigger, market_state, regime, trade_signal, executed, outcome, reasoning_output, embedding, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                params,
            )
        return self._insert(
            "INSERT INTO episodes (timestamp, actor, trigger, market_state, regime, trade_signal, executed, outcome, reasoning_output, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params,
        )

    def query_episodes(
        self,
        actor: Optional[str] = None,
        regime: Optional[str] = None,
        outcome: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM episodes WHERE 1=1"
        params: list[Any] = []
        if actor:
            sql += " AND actor = %s" if self.is_postgres else " AND actor = ?"
            params.append(actor)
        if regime:
            sql += " AND regime = %s" if self.is_postgres else " AND regime = ?"
            params.append(regime)
        if outcome:
            sql += " AND outcome LIKE %s" if self.is_postgres else " AND outcome LIKE ?"
            params.append(f'%"outcome": "{outcome}"%')
        if since:
            sql += " AND timestamp >= %s" if self.is_postgres else " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp DESC LIMIT %s" if self.is_postgres else " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.fetchall(sql, tuple(params))
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["market_state"] = self._from_json(d.get("market_state"))
            d["trade_signal"] = self._from_json(d.get("trade_signal"))
            d["outcome"] = self._from_json(d.get("outcome"))
            d["reasoning_output"] = self._from_json(d.get("reasoning_output"))
            d["embedding"] = self._from_json(d.get("embedding"))
            d["executed"] = bool(d.get("executed", 1))
            out.append(d)
        return out

    def search_episodes(self, query_embedding: list[float], top_k: int = 10) -> list[dict[str, Any]]:
        if self.is_postgres:
            rows = self.fetchall(
                "SELECT *, embedding <=> %s::vector AS distance FROM episodes ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_embedding, query_embedding, top_k),
            )
            out = []
            for r in rows:
                d = self._row_to_dict(r)
                d["market_state"] = self._from_json(d.get("market_state"))
                d["trade_signal"] = self._from_json(d.get("trade_signal"))
                d["outcome"] = self._from_json(d.get("outcome"))
                d["reasoning_output"] = self._from_json(d.get("reasoning_output"))
                d["embedding"] = self._from_json(d.get("embedding"))
                out.append(d)
            return out
        rows = self.fetchall("SELECT * FROM episodes")
        scored = []
        for r in rows:
            emb = self._from_json(r["embedding"])
            if emb:
                scored.append((self._cosine_similarity(query_embedding, emb), self._row_to_dict(r)))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for _, item in scored[:top_k]:
            item["market_state"] = self._from_json(item.get("market_state"))
            item["trade_signal"] = self._from_json(item.get("trade_signal"))
            item["outcome"] = self._from_json(item.get("outcome"))
            item["reasoning_output"] = self._from_json(item.get("reasoning_output"))
            item["embedding"] = self._from_json(item.get("embedding"))
            result.append(item)
        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (norm_a * norm_b)

    # Conscious summaries ----------------------------------------------------

    def insert_conscious_summary(self, summary: str, context: Optional[dict[str, Any]] = None) -> int:
        return self._insert(
            "INSERT INTO conscious_summaries (summary, context, created_at) VALUES (%s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO conscious_summaries (summary, context, created_at) VALUES (?, ?, ?)",
            (summary, self._to_json(context), self._now()),
        )

    def recent_conscious_summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT * FROM conscious_summaries ORDER BY created_at DESC LIMIT %s"
            if self.is_postgres
            else "SELECT * FROM conscious_summaries ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["context"] = self._from_json(d.get("context"))
            out.append(d)
        return out

    def insert_consciousness_report(self, report: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO consciousness_reports (period, report_date, report_text, score, metadata, created_at) VALUES (%s, %s, %s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO consciousness_reports (period, report_date, report_text, score, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                report["period"],
                report["report_date"],
                report["report_text"],
                report.get("score"),
                self._to_json(report.get("metadata")),
                self._now(),
            ),
        )

    def get_consciousness_report(self, period: str, report_date: str) -> Optional[dict[str, Any]]:
        row = self.fetchone(
            "SELECT * FROM consciousness_reports WHERE period = %s AND report_date = %s ORDER BY created_at DESC LIMIT 1"
            if self.is_postgres
            else "SELECT * FROM consciousness_reports WHERE period = ? AND report_date = ? ORDER BY created_at DESC LIMIT 1",
            (period, report_date),
        )
        if not row:
            return None
        d = self._row_to_dict(row)
        d["metadata"] = self._from_json(d.get("metadata"))
        return d

    def query_consciousness_reports(self, period: Optional[str] = None, limit: int = 30) -> list[dict[str, Any]]:
        sql = "SELECT * FROM consciousness_reports WHERE 1=1"
        params: list[Any] = []
        if period:
            sql += " AND period = %s" if self.is_postgres else " AND period = ?"
            params.append(period)
        sql += " ORDER BY created_at DESC LIMIT %s" if self.is_postgres else " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.fetchall(sql, tuple(params))
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["metadata"] = self._from_json(d.get("metadata"))
            out.append(d)
        return out

    # Events -----------------------------------------------------------------

    def insert_event(self, event: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO events (ts, level, category, phase, message, symbol, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO events (ts, level, category, phase, message, symbol, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.get("ts") or self._now(),
                event.get("level", "INFO"),
                event["category"],
                event.get("phase"),
                event["message"],
                event.get("symbol"),
                self._to_json(event.get("metadata")),
            ),
        )

    def recent_events(
        self,
        limit: int = 100,
        category: Optional[str] = None,
        after_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if category:
            sql += " AND category = %s" if self.is_postgres else " AND category = ?"
            params.append(category)
        if after_id:
            sql += " AND id > %s" if self.is_postgres else " AND id > ?"
            params.append(after_id)
        sql += " ORDER BY ts DESC LIMIT %s" if self.is_postgres else " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self.fetchall(sql, tuple(params))
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["metadata"] = self._from_json(d.get("metadata"))
            out.append(d)
        return out

    # Versions ---------------------------------------------------------------

    def insert_version(self, version: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO versions (module, version, path, diff, new_content, status, created_at) VALUES (%s, %s, %s, %s, %s, 'proposed', %s)"
            if self.is_postgres
            else "INSERT INTO versions (module, version, path, diff, new_content, status, created_at) VALUES (?, ?, ?, ?, ?, 'proposed', ?)",
            (version["module"], version["version"], version.get("path"), version.get("diff"), version.get("new_content"), self._now()),
        )

    def list_versions(self, module: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if module:
            rows = self.fetchall(
                "SELECT * FROM versions WHERE module = %s ORDER BY created_at DESC LIMIT %s"
                if self.is_postgres
                else "SELECT * FROM versions WHERE module = ? ORDER BY created_at DESC LIMIT ?",
                (module, limit),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM versions ORDER BY created_at DESC LIMIT %s"
                if self.is_postgres
                else "SELECT * FROM versions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [self._row_to_dict(r) for r in rows]

    def update_version_status(self, version_id: int, status: str) -> None:
        self.execute(
            "UPDATE versions SET status = %s, applied_at = %s WHERE id = %s"
            if self.is_postgres
            else "UPDATE versions SET status = ?, applied_at = ? WHERE id = ?",
            (status, self._now() if status == "applied" else None, version_id),
        )

    # Audit log --------------------------------------------------------------

    def audit(self, actor: str, action: str, detail: Optional[dict[str, Any]] = None) -> int:
        return self._insert(
            "INSERT INTO audit_log (actor, action, detail, created_at) VALUES (%s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO audit_log (actor, action, detail, created_at) VALUES (?, ?, ?, ?)",
            (actor, action, self._to_json(detail), self._now()),
        )

    def recent_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT %s"
            if self.is_postgres
            else "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["detail"] = self._from_json(d.get("detail"))
            out.append(d)
        return out

    # Conversations ----------------------------------------------------------

    def add_message(self, role: str, message: str, metadata: Optional[dict[str, Any]] = None) -> int:
        return self._insert(
            "INSERT INTO conversations (role, message, metadata, ts) VALUES (%s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO conversations (role, message, metadata, ts) VALUES (?, ?, ?, ?)",
            (role, message, self._to_json(metadata), self._now()),
        )

    def recent_messages(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT * FROM conversations ORDER BY ts DESC LIMIT %s"
            if self.is_postgres
            else "SELECT * FROM conversations ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["metadata"] = self._from_json(d.get("metadata"))
            out.append(d)
        return out

    # Brain cycles -----------------------------------------------------------

    def start_cycle(self) -> int:
        return self._insert(
            "INSERT INTO brain_cycles (started_at, status) VALUES (%s, 'running')"
            if self.is_postgres
            else "INSERT INTO brain_cycles (started_at, status) VALUES (?, 'running')",
            (self._now(),),
        )

    def finish_cycle(self, cycle_id: int, summary: str, status: str = "ok") -> None:
        self.execute(
            "UPDATE brain_cycles SET finished_at = %s, status = %s, summary = %s WHERE id = %s"
            if self.is_postgres
            else "UPDATE brain_cycles SET finished_at = ?, status = ?, summary = ? WHERE id = ?",
            (self._now(), status, summary, cycle_id),
        )

    def last_cycle(self) -> Optional[dict[str, Any]]:
        row = self.fetchone(
            "SELECT * FROM brain_cycles ORDER BY started_at DESC LIMIT 1"
        )
        return self._row_to_dict(row) if row else None

    # Reports ----------------------------------------------------------------

    def insert_report(self, report: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO reports (slug, title, category, report_date, description, content_json, status, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO reports (slug, title, category, report_date, description, content_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report["slug"],
                report["title"],
                report["category"],
                report["report_date"],
                report.get("description"),
                self._to_json(report.get("content_json", {})),
                report.get("status", "published"),
                self._now(),
                self._now(),
            ),
        )

    def get_report(self, slug: str) -> Optional[dict[str, Any]]:
        row = self.fetchone(
            "SELECT * FROM reports WHERE slug = %s" if self.is_postgres else "SELECT * FROM reports WHERE slug = ?",
            (slug,),
        )
        if row:
            d = self._row_to_dict(row)
            d["content_json"] = self._from_json(d.get("content_json"))
            return d
        return None

    def list_reports(self, category: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        if category:
            rows = self.fetchall(
                "SELECT * FROM reports WHERE category = %s ORDER BY report_date DESC LIMIT %s"
                if self.is_postgres
                else "SELECT * FROM reports WHERE category = ? ORDER BY report_date DESC LIMIT ?",
                (category, limit),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM reports ORDER BY report_date DESC LIMIT %s"
                if self.is_postgres
                else "SELECT * FROM reports ORDER BY report_date DESC LIMIT ?",
                (limit,),
            )
        out = []
        for r in rows:
            d = self._row_to_dict(r)
            d["content_json"] = self._from_json(d.get("content_json"))
            out.append(d)
        return out

    def upsert_report(self, report: dict[str, Any]) -> int:
        existing = self.get_report(report["slug"])
        if existing:
            self.execute(
                "UPDATE reports SET title = %s, category = %s, report_date = %s, description = %s, content_json = %s, status = %s, updated_at = %s WHERE slug = %s"
                if self.is_postgres
                else "UPDATE reports SET title = ?, category = ?, report_date = ?, description = ?, content_json = ?, status = ?, updated_at = ? WHERE slug = ?",
                (
                    report["title"],
                    report["category"],
                    report["report_date"],
                    report.get("description"),
                    self._to_json(report.get("content_json", {})),
                    report.get("status", "published"),
                    self._now(),
                    report["slug"],
                ),
            )
            return existing["id"]
        return self.insert_report(report)

    # Market deltas ----------------------------------------------------------

    def insert_market_delta(self, delta: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO market_deltas (symbol, price, change_pct, category, report_date, created_at) VALUES (%s, %s, %s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO market_deltas (symbol, price, change_pct, category, report_date, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                delta["symbol"],
                delta["price"],
                delta["change_pct"],
                delta["category"],
                delta["report_date"],
                self._now(),
            ),
        )

    def get_market_deltas(self, report_date: str, category: Optional[str] = None) -> list[dict[str, Any]]:
        if category:
            rows = self.fetchall(
                "SELECT * FROM market_deltas WHERE report_date = %s AND category = %s ORDER BY symbol"
                if self.is_postgres
                else "SELECT * FROM market_deltas WHERE report_date = ? AND category = ? ORDER BY symbol",
                (report_date, category),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM market_deltas WHERE report_date = %s ORDER BY category, symbol"
                if self.is_postgres
                else "SELECT * FROM market_deltas WHERE report_date = ? ORDER BY category, symbol",
                (report_date,),
            )
        return [self._row_to_dict(r) for r in rows]

    # Scheduled events -------------------------------------------------------

    def insert_scheduled_event(self, event: dict[str, Any]) -> int:
        tags = event.get("tags")
        if not self.is_postgres and tags is not None:
            tags = self._to_json(tags)
        return self._insert(
            "INSERT INTO scheduled_events (title, category, impact, event_time, description, tags, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO scheduled_events (title, category, impact, event_time, description, tags, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event["title"],
                event["category"],
                event.get("impact", "Medium"),
                event["event_time"],
                event.get("description"),
                tags,
                self._now(),
            ),
        )

    def list_scheduled_events(
        self,
        after: Optional[str] = None,
        before: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM scheduled_events WHERE 1=1"
        params: list[Any] = []
        if after:
            sql += " AND event_time >= %s" if self.is_postgres else " AND event_time >= ?"
            params.append(after)
        if before:
            sql += " AND event_time <= %s" if self.is_postgres else " AND event_time <= ?"
            params.append(before)
        if category:
            sql += " AND category = %s" if self.is_postgres else " AND category = ?"
            params.append(category)
        sql += " ORDER BY event_time LIMIT %s" if self.is_postgres else " ORDER BY event_time LIMIT ?"
        params.append(limit)
        out = []
        for r in self.fetchall(sql, tuple(params)):
            d = self._row_to_dict(r)
            d["tags"] = self._from_json(d.get("tags"))
            out.append(d)
        return out

    # Inventory entries ------------------------------------------------------

    def insert_inventory_entry(self, entry: dict[str, Any]) -> int:
        return self._insert(
            "INSERT INTO inventory_entries (source, rel_path, name, entry_type, size_bytes, content_hash, content, metadata, assimilation_score, assimilation_status, assimilation_reason, version_id, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            if self.is_postgres
            else "INSERT INTO inventory_entries (source, rel_path, name, entry_type, size_bytes, content_hash, content, metadata, assimilation_score, assimilation_status, assimilation_reason, version_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry["source"],
                entry["rel_path"],
                entry.get("name"),
                entry["entry_type"],
                entry.get("size_bytes"),
                entry.get("content_hash"),
                entry.get("content"),
                self._to_json(entry.get("metadata")),
                entry.get("assimilation_score"),
                entry.get("assimilation_status", "pending"),
                entry.get("assimilation_reason"),
                entry.get("version_id"),
                self._now(),
                self._now(),
            ),
        )

    def get_inventory_entry(self, entry_id: int) -> Optional[dict[str, Any]]:
        row = self.fetchone(
            "SELECT * FROM inventory_entries WHERE id = %s" if self.is_postgres else "SELECT * FROM inventory_entries WHERE id = ?",
            (entry_id,),
        )
        if not row:
            return None
        d = self._row_to_dict(row)
        d["metadata"] = self._from_json(d.get("metadata"))
        return d

    def get_inventory_entries(
        self,
        source: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM inventory_entries WHERE 1=1"
        params: list[Any] = []
        if source:
            sql += " AND source = %s" if self.is_postgres else " AND source = ?"
            params.append(source)
        if status:
            sql += " AND assimilation_status = %s" if self.is_postgres else " AND assimilation_status = ?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT %s OFFSET %s" if self.is_postgres else " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        out = []
        for r in self.fetchall(sql, tuple(params)):
            d = self._row_to_dict(r)
            d["metadata"] = self._from_json(d.get("metadata"))
            out.append(d)
        return out

    def update_inventory_status(
        self,
        entry_id: int,
        status: str,
        reason: Optional[str] = None,
        score: Optional[float] = None,
        version_id: Optional[int] = None,
    ) -> None:
        fields = ["assimilation_status = %s" if self.is_postgres else "assimilation_status = ?"]
        params: list[Any] = [status]
        if reason is not None:
            fields.append("assimilation_reason = %s" if self.is_postgres else "assimilation_reason = ?")
            params.append(reason)
        if score is not None:
            fields.append("assimilation_score = %s" if self.is_postgres else "assimilation_score = ?")
            params.append(score)
        if version_id is not None:
            fields.append("version_id = %s" if self.is_postgres else "version_id = ?")
            params.append(version_id)
        fields.append("updated_at = %s" if self.is_postgres else "updated_at = ?")
        params.append(self._now())
        params.append(entry_id)
        sql = "UPDATE inventory_entries SET " + ", ".join(fields) + " WHERE id = %s" if self.is_postgres else "UPDATE inventory_entries SET " + ", ".join(fields) + " WHERE id = ?"
        self.execute(sql, tuple(params))


db = Database()
