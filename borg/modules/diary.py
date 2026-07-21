"""Daily diary writer for Borg.

Generates a Markdown summary of each day's activity — forecasts, HIP-4
predictions, paper trades, learnings, reflections, events, and system status —
and writes it to a configurable output folder.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from borg.config import settings
from borg.db import Database, db
from borg.modules.strategy_report import generate_strategy_report


class DiaryWriter:
    """Write a daily Markdown diary from database state."""

    def __init__(
        self,
        database: Optional[Database] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.db = database or db
        self.output_dir = output_dir or settings.diary_output_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _today(self) -> date:
        return datetime.now(timezone.utc).date()

    def _format_dt(self, value: Any) -> str:
        if value is None:
            return "--"
        if isinstance(value, datetime):
            return value.replace(microsecond=0).isoformat()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(microsecond=0).isoformat()
            except Exception:
                return value
        return str(value)

    def _section(self, title: str, body: str) -> str:
        return f"## {title}\n\n{body}\n\n"

    def _market_summary(self, report_date: date) -> str:
        start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        ph = self.db.placeholder
        candles = self.db.fetchall(
            f"SELECT symbol, close, ts FROM market_candles WHERE ts >= {ph} AND ts < {ph} ORDER BY ts DESC",
            (start.isoformat(), end.isoformat()),
        )
        if not candles:
            return "No market candles recorded today.\n"
        latest_by_symbol: dict[str, Any] = {}
        for row in candles:
            sym = row["symbol"]
            if sym not in latest_by_symbol:
                latest_by_symbol[sym] = row
        lines = ["| Symbol | Price | Time |", "|--------|-------|------|"]
        for sym in sorted(latest_by_symbol):
            row = latest_by_symbol[sym]
            lines.append(f"| {sym} | {row['close']} | {self._format_dt(row['ts'])[11:19]} |")
        return "\n".join(lines) + "\n"

    def _forecasts_summary(self, report_date: date) -> str:
        start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        ph = self.db.placeholder
        rows = self.db.fetchall(
            f"SELECT symbol, direction, confidence, outcome, correct, created_at FROM forecasts WHERE created_at >= {ph} AND created_at < {ph} ORDER BY created_at DESC",
            (start.isoformat(), end.isoformat()),
        )
        if not rows:
            return "No forecasts recorded today.\n"
        lines = ["| Time | Symbol | Direction | Confidence | Outcome |", "|------|--------|-----------|------------|---------|"]
        for row in rows:
            ts = self._format_dt(row["created_at"])[11:19]
            outcome = row["outcome"] or "pending"
            lines.append(
                f"| {ts} | {row['symbol']} | {row['direction'].upper()} | {row['confidence']:.1f}% | {outcome} |"
            )
        return "\n".join(lines) + "\n"

    def _hip4_summary(self, report_date: date) -> str:
        # hip4_predictions does not have a created_at column; filter by expiry date.
        start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        ph = self.db.placeholder
        rows = self.db.fetchall(
            f"SELECT underlying, target_price, direction, confidence, expiry, yes_price, no_price FROM hip4_predictions WHERE expiry >= {ph} AND expiry < {ph} ORDER BY underlying",
            (start.isoformat(), end.isoformat()),
        )
        if not rows:
            return "No HIP-4 predictions recorded today.\n"
        lines = ["| Underlying | Strike | Direction | Confidence | Yes | No |", "|------------|--------|-----------|------------|-----|-----|"]
        for row in rows:
            lines.append(
                f"| {row['underlying']} | {row['target_price']} | {row['direction'].upper()} | {row['confidence']:.1f}% | {row['yes_price']} | {row['no_price']} |"
            )
        return "\n".join(lines) + "\n"

    def _paper_trades_summary(self, report_date: date) -> str:
        date_str = report_date.isoformat()
        ph = self.db.placeholder
        rows = self.db.fetchall(
            f"SELECT underlying, side, target_price, token_price, quantity, stake, potential_payout, outcome, settle_price, pnl, expiry FROM paper_trades WHERE trade_date = {ph} ORDER BY trade_date DESC",
            (date_str,),
        )
        if not rows:
            return "No paper trades recorded for today.\n"
        lines = ["| Underlying | Side | Strike | Token Price | Quantity | Stake | Outcome | PnL |", "|------------|------|--------|-------------|----------|-------|---------|-----|"]
        for row in rows:
            pnl = f"{row['pnl']:.4f}" if row["pnl"] is not None else "--"
            lines.append(
                f"| {row['underlying']} | {row['side']} | {row['target_price']} | {row['token_price']} | {row['quantity']:.4f} | ${row['stake']:.2f} | {row['outcome']} | {pnl} |"
            )
        return "\n".join(lines) + "\n"

    def _learnings_summary(self, report_date: date) -> str:
        start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        ph = self.db.placeholder
        rows = self.db.fetchall(
            f"SELECT summary, detail, tags, created_at FROM learnings WHERE created_at >= {ph} AND created_at < {ph} ORDER BY created_at DESC LIMIT 20",
            (start.isoformat(), end.isoformat()),
        )
        if not rows:
            return "No learnings recorded today.\n"
        lines = []
        for row in rows:
            ts = self._format_dt(row["created_at"])[11:19]
            tags = row["tags"] or ""
            lines.append(f"- **{ts}** — {row['summary']}  ")
            if row["detail"]:
                lines.append(f"  Detail: {row['detail'][:200]}")
            if tags:
                lines.append(f"  Tags: `{tags}`")
        return "\n".join(lines) + "\n"

    def _reflections_summary(self, report_date: date) -> str:
        start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        ph = self.db.placeholder
        rows = self.db.fetchall(
            f"SELECT critique, score, created_at FROM reflections WHERE created_at >= {ph} AND created_at < {ph} ORDER BY created_at DESC LIMIT 20",
            (start.isoformat(), end.isoformat()),
        )
        if not rows:
            return "No reflections recorded today.\n"
        lines = []
        for row in rows:
            ts = self._format_dt(row["created_at"])[11:19]
            lines.append(f"- **{ts}** — score {row['score']:.2f}: {row['critique']}")
        return "\n".join(lines) + "\n"

    def _events_summary(self, report_date: date) -> str:
        start = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        ph = self.db.placeholder
        rows = self.db.fetchall(
            f"SELECT ts, level, category, phase, message, symbol FROM events WHERE ts >= {ph} AND ts < {ph} ORDER BY ts DESC LIMIT 50",
            (start.isoformat(), end.isoformat()),
        )
        if not rows:
            return "No events recorded today.\n"
        lines = []
        for row in rows:
            ts = self._format_dt(row["ts"])[11:19]
            sym = f" [{row['symbol']}]" if row["symbol"] else ""
            lines.append(f"- **{ts}** `{row['level']}` [{row['category']}/{row['phase']}] {row['message']}{sym}")
        return "\n".join(lines) + "\n"

    def _system_status(self) -> str:
        from borg.monitor import monitor

        status = monitor.status()
        lines = [
            f"- CPU: {status.cpu_percent:.1f}%",
            f"- Memory: {status.memory_used_mb:.1f} / {status.memory_total_mb:.1f} MB",
            f"- Active symbols: {', '.join(status.active_symbols)}",
            f"- Database: {status.db_path}",
        ]
        return "\n".join(lines) + "\n"

    def _strategy_report(self) -> str:
        report = generate_strategy_report(self.db)
        if not report["total_forecasts"]:
            return "No strategy signals recorded today.\n"
        lines = [f"- Total forecasts: {report['total_forecasts']}", ""]
        lines.append("**By Symbol (forecast direction)**")
        lines.append("| Symbol | Up | Down | Flat |")
        lines.append("|--------|----|------|------|")
        for sym, counts in sorted(report["by_symbol"].items()):
            lines.append(
                f"| {sym} | {counts.get('up', 0)} | {counts.get('down', 0)} | {counts.get('flat', 0)} |"
            )
        lines.append("")
        lines.append("**By Strategy (signal action)**")
        lines.append("| Strategy | Buy | Sell | Hold |")
        lines.append("|----------|-----|------|------|")
        for name, counts in sorted(report["by_strategy"].items()):
            lines.append(f"| {name} | {counts['buy']} | {counts['sell']} | {counts['hold']} |")
        return "\n".join(lines) + "\n"

    def _hyperlong_summary(self, hyperlong_data: Optional[dict[str, Any]]) -> str:
        """Format a HyperLong indicator snapshot for the diary."""
        if not hyperlong_data:
            return "No HyperLong data available.\n"
        lines = ["| Symbol | Price | EMA100 | RSI14 | ADX14 | Labels |", "|--------|-------|--------|-------|-------|--------|"]
        for symbol, data in sorted(hyperlong_data.items()):
            if not isinstance(data, dict) or data.get("error"):
                lines.append(f"| {symbol} | error | -- | -- | -- | -- |")
                continue
            price = self._last_value(data.get("price", data.get("close")))
            ema100 = self._last_value(data.get("ema100"))
            rsi14 = self._last_value(data.get("rsi14"))
            adx14 = self._last_value(data.get("adx14"))
            labels = ", ".join(str(l) for l in data.get("labels", [])[:3]) or "--"
            lines.append(f"| {symbol} | {price} | {ema100} | {rsi14} | {adx14} | {labels} |")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _last_value(series: Any) -> str:
        """Return the last value of a list/series or a placeholder."""
        if isinstance(series, (list, tuple)) and series:
            return str(series[-1])
        if series is None:
            return "--"
        return str(series)

    def write_daily_diary(
        self,
        report_date: Optional[date] = None,
        hyperlong_data: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Generate and write the daily diary Markdown file."""
        report_date = report_date or self._today()
        now = datetime.now(timezone.utc)

        lines: list[str] = [
            f"# Borg Daily Diary — {report_date.isoformat()}",
            "",
            f"Generated: {now.replace(microsecond=0).isoformat()} UTC",
            "",
        ]
        body = (
            self._section("Market Snapshot", self._market_summary(report_date))
            + self._section("HyperLong Snapshot", self._hyperlong_summary(hyperlong_data))
            + self._section("Forecasts", self._forecasts_summary(report_date))
            + self._section("Strategy Report", self._strategy_report())
            + self._section("HIP-4 Predictions", self._hip4_summary(report_date))
            + self._section("Paper Trades", self._paper_trades_summary(report_date))
            + self._section("Learnings", self._learnings_summary(report_date))
            + self._section("Reflections", self._reflections_summary(report_date))
            + self._section("Events", self._events_summary(report_date))
            + self._section("System Status", self._system_status())
        )
        lines.append(body)

        path = self.output_dir / f"{report_date.isoformat()}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


# Module-level singleton
diary_writer = DiaryWriter()


def list_diary_files(output_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Return metadata for all diary files in the output directory."""
    directory = output_dir or settings.diary_output_path
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.md"), reverse=True)
    return [
        {
            "date": f.stem,
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "updated_at": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        for f in files
    ]
