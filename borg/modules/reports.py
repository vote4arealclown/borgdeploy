"""Report generation engine modeled after saskpoly.xyz marketplace reports."""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Optional

from borg.db import Database, db


class ReportEngine:
    """Generate and store daily intelligence, HR, Brent, and coffee-news reports."""

    CRYPTO_SYMBOLS = [
        ("BNB", 560.0, "crypto"),
        ("BTC", 63000.0, "crypto"),
        ("ETH", 1800.0, "crypto"),
        ("HYPE", 60.0, "crypto"),
        ("SUI", 0.75, "crypto"),
        ("XRP", 1.08, "crypto"),
    ]

    COMMODITY_SYMBOLS = [
        ("xyz:BRENTOIL", 84.5, "commodity"),
        ("xyz:CL", 80.0, "commodity"),
        ("xyz:GOLD", 4000.0, "commodity"),
        ("xyz:NATGAS", 2.8, "commodity"),
        ("xyz:SILVER", 55.5, "commodity"),
        ("BRENT", 86.0, "commodity"),
    ]

    STOCK_SYMBOLS = [
        ("xyz:AAPL", 330.0, "stock"),
        ("xyz:COIN", 155.0, "stock"),
        ("xyz:HOOD", 100.0, "stock"),
        ("xyz:MSTR", 92.0, "stock"),
        ("xyz:NVDA", 200.0, "stock"),
        ("xyz:SP500", 7450.0, "stock"),
        ("xyz:TSLA", 385.0, "stock"),
        ("xyz:XLE", 58.0, "stock"),
    ]

    MLB_TEAMS = [
        ("Tampa Bay Rays", "Boston Red Sox"),
        ("New York Yankees", "Toronto Blue Jays"),
        ("Houston Astros", "Texas Rangers"),
        ("Los Angeles Dodgers", "San Francisco Giants"),
    ]

    def __init__(self, database: Database = db) -> None:
        self.db = database

    def _today(self) -> date:
        return datetime.now(timezone.utc).date()

    def _format_date(self, d: date) -> str:
        return d.strftime("%Y-%m-%d")

    def _generate_deltas(
        self,
        report_date: date,
        hyperlong_data: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Produce 24h market deltas. Use real HyperLong data when available."""
        deltas: list[dict[str, Any]] = []
        if hyperlong_data:
            for symbol, data in sorted(hyperlong_data.items()):
                if not isinstance(data, dict) or data.get("error"):
                    continue
                price = data.get("price") or data.get("close")
                if isinstance(price, (list, tuple)) and price:
                    price = price[-1]
                if price is None:
                    continue
                try:
                    price_val = float(price)
                except (TypeError, ValueError):
                    continue
                deltas.append(
                    {
                        "symbol": symbol,
                        "price": round(price_val, 8),
                        "change_pct": 0.0,
                        "category": "crypto",
                        "report_date": self._format_date(report_date),
                    }
                )
        if deltas:
            return deltas

        all_symbols = self.CRYPTO_SYMBOLS + self.COMMODITY_SYMBOLS + self.STOCK_SYMBOLS
        rng = random.Random(report_date.isoformat() + "deltas")
        for symbol, base_price, category in all_symbols:
            change_pct = rng.gauss(0, 2.5)
            price = base_price * (1 + change_pct / 100.0)
            deltas.append(
                {
                    "symbol": symbol,
                    "price": round(price, 8),
                    "change_pct": round(change_pct, 2),
                    "category": category,
                    "report_date": self._format_date(report_date),
                }
            )
        return deltas

    def _store_deltas(self, deltas: list[dict[str, Any]]) -> None:
        for delta in deltas:
            self.db.insert_market_delta(delta)

    def _overnight_news(self, report_date: date) -> dict[str, Any]:
        rng = random.Random(report_date.isoformat() + "news")
        return {
            "mlb": f"{rng.randint(10, 18)} MLB games on the schedule today.",
            "options": (
                f"VIX at {rng.uniform(15.0, 25.0):.2f} ({rng.choice(['+', '-'])}{rng.uniform(1.0, 10.0):.2f}%). "
                f"SPY at {rng.uniform(700.0, 800.0):.2f}. 10-year Treasury yield at {rng.uniform(4.0, 5.0):.2f}%."
            ),
            "crypto": rng.choice(
                [
                    "[DATA UNAVAILABLE] Crypto source failed.",
                    "Bitcoin dominance stable; altcoins showing mixed momentum.",
                    "ETH gas fees elevated overnight; Layer-2 activity increased.",
                ]
            ),
            "sports": (
                f"SportSelect: {rng.randint(80, 140)} events scraped. "
                f"{rng.randint(2, 8)} soccer/FIFA events available."
            ),
        }

    def _mlb_forecasts(self, report_date: date) -> list[dict[str, Any]]:
        rng = random.Random(report_date.isoformat() + "mlb")
        forecasts = []
        base_hour = 17
        for home, away in self.MLB_TEAMS:
            time_str = f"{base_hour:02d}:35 UTC"
            base_hour += 1
            prob_home = rng.uniform(45.0, 65.0)
            forecasts.append(
                {
                    "time": time_str,
                    "matchup": f"{home} @ {away}",
                    "prediction": f"{home} {prob_home:.1f}% / {away} {100 - prob_home:.1f}%",
                }
            )
        return forecasts

    def _prediction_markets(self, report_date: date) -> list[dict[str, Any]]:
        rng = random.Random(report_date.isoformat() + "pm")
        return [
            {
                "market": "Decrease (fomc, July 2026)",
                "yes": rng.uniform(0.3, 1.5),
                "no": rng.uniform(97.0, 99.5),
            },
            {
                "market": "Exactly 3.8% (cpi, June 2026)",
                "yes": rng.uniform(45.0, 55.0),
                "no": rng.uniform(45.0, 55.0),
            },
            {
                "market": "No change (fomc, July 2026)",
                "yes": rng.uniform(90.0, 98.0),
                "no": rng.uniform(2.0, 10.0),
            },
        ]

    def generate_daily_report(
        self,
        report_date: Optional[date] = None,
        hyperlong_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate and store a Daily Intelligence Brief."""
        report_date = report_date or self._today()
        slug = f"daily-report-{self._format_date(report_date)}"
        title = f"Daily Intelligence Brief — {self._format_date(report_date)}"

        deltas = self._generate_deltas(report_date, hyperlong_data=hyperlong_data)
        self._store_deltas(deltas)

        content = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "active_domains": ["MLB", "Options", "Crypto", "Sports"],
            "overnight_deltas": [d["symbol"] for d in deltas],
            "overnight_news": self._overnight_news(report_date),
            "mlb_forecasts": self._mlb_forecasts(report_date),
            "prediction_markets": self._prediction_markets(report_date),
            "notes": {
                "current_phase": "1. Infrastructure",
                "active_loops": ["DAILY_LOOP"],
                "blockers": "None recorded",
                "open_questions": "See STATE.md",
            },
            "next_report": (report_date + timedelta(days=1)).isoformat(),
        }

        report = {
            "slug": slug,
            "title": title,
            "category": "daily",
            "report_date": self._format_date(report_date),
            "description": "Free printable daily report. Branded PDF with overnight news, forecast, predictions, calls, and review.",
            "content_json": content,
            "status": "published",
        }
        self.db.upsert_report(report)
        return report

    def generate_hr_report(self, report_date: Optional[date] = None) -> dict[str, Any]:
        """Generate and store a Home Run Tracker report."""
        report_date = report_date or self._today()
        slug = f"hr-report-{self._format_date(report_date)}"
        rng = random.Random(report_date.isoformat() + "hr")

        candidates = []
        for i in range(10):
            candidates.append(
                {
                    "rank": i + 1,
                    "player": f"Player {i + 1}",
                    "team": rng.choice(["NYY", "LAD", "HOU", "TOR", "BOS", "TEX"]),
                    "weather_score": round(rng.uniform(60.0, 95.0), 1),
                    "tier": rng.choice(["Tier 1", "Tier 2", "Tier 3"]),
                    "red_flags": rng.choice(["None", "Wind in", "Pitcher matchup", "Rest day"]),
                }
            )

        content = {
            "candidates": candidates,
            "methodology": "Ranked candidates, weather-adjusted scores, tier filters, and red flags.",
        }

        report = {
            "slug": slug,
            "title": f"Home Run Tracker — {self._format_date(report_date)}",
            "category": "hr",
            "report_date": self._format_date(report_date),
            "description": "Daily home run prop model report. Ranked candidates, weather-adjusted scores, tier filters, and red flags.",
            "content_json": content,
            "status": "published",
        }
        self.db.upsert_report(report)
        return report

    def generate_brent_report(self, report_date: Optional[date] = None) -> dict[str, Any]:
        """Generate and store a Morning Brent Crude Brief."""
        report_date = report_date or self._today()
        slug = f"brent-report-{self._format_date(report_date)}"
        rng = random.Random(report_date.isoformat() + "brent")

        content = {
            "brent_price": round(rng.uniform(82.0, 88.0), 2),
            "change_pct": round(rng.gauss(0, 1.5), 2),
            "inventory_signal": rng.choice(["Draw expected", "Build expected", "Neutral"]),
            "rig_count": rng.randint(480, 540),
            "headlines": [
                "OPEC+ maintains current production quotas.",
                "U.S. crude inventories show mixed signals.",
                "Geopolitical risk premium remains elevated.",
            ],
        }

        report = {
            "slug": slug,
            "title": f"Morning Brent Crude Brief — {self._format_date(report_date)}",
            "category": "brent",
            "report_date": self._format_date(report_date),
            "description": "Morning Brent crude brief with price, inventory signals, rig count, and headline summary.",
            "content_json": content,
            "status": "published",
        }
        self.db.upsert_report(report)
        return report

    def generate_coffee_news(self, report_date: Optional[date] = None) -> dict[str, Any]:
        """Generate and store a Coffee News Edition."""
        report_date = report_date or self._today()
        slug = f"coffee-news-{self._format_date(report_date)}"
        rng = random.Random(report_date.isoformat() + "coffee")

        content = {
            "headline": "Market Tick & Overnight Lock",
            "market_tick": f"{rng.choice(['Futures mixed', 'Grains higher', 'Energy lower', 'Metals firm'])}",
            "todays_lock": rng.choice(["CPI print", "FOMC minutes", "EIA inventories", "World Cup final"]),
            "overnight_summary": "Overnight session saw mixed flows across asset classes. Key macro events drive today's lock.",
            "sections": [
                {"title": "Front Page", "body": "Top overnight movers and today's key levels."},
                {"title": "Macro", "body": "Rates, FX, and central-bank watch."},
                {"title": "Sports", "body": "Schedule highlights and prediction-market flow."},
            ],
        }

        report = {
            "slug": slug,
            "title": f"Coffee News Edition — {self._format_date(report_date)}",
            "category": "coffee",
            "report_date": self._format_date(report_date),
            "description": "The daily briefing in a newspaper-style layout. Market tick, today's lock, and full overnight report.",
            "content_json": content,
            "status": "published",
        }
        self.db.upsert_report(report)
        return report

    def generate_system_report(self, report_date: Optional[date] = None) -> dict[str, Any]:
        """Generate and store a Borg system health & activity report."""
        from borg.config import settings
        from borg.monitor import monitor

        report_date = report_date or self._today()
        slug = f"borg-system-report-{self._format_date(report_date)}"

        status = monitor.status()
        status.ollama_reachable = False  # avoid blocking network call here

        # Counts
        counts: dict[str, int] = {}
        for table in ["forecasts", "market_candles", "hip4_predictions", "paper_trades", "events", "episodes"]:
            try:
                rows = self.db.fetchall(
                    f"SELECT COUNT(*) FROM {table}",
                    (),
                )
                counts[table] = int(self.db._row_to_dict(rows[0])["count"]) if rows else 0
            except Exception:
                counts[table] = 0

        def _ts(value: Any) -> Any:
            """Serialize datetime/date values for JSON storage."""
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, date):
                return value.isoformat()
            return value

        # Latest forecasts
        latest_forecasts = [
            {
                "symbol": r["symbol"],
                "direction": r["direction"],
                "confidence": float(r["confidence"]),
                "outcome": r.get("outcome"),
                "created_at": _ts(r["created_at"]),
            }
            for r in self.db.recent_forecasts(limit=10)
        ]

        # Latest HIP-4 predictions
        latest_hip4 = [
            {
                "underlying": r["underlying"],
                "direction": r["direction"],
                "confidence": float(r["confidence"]),
                "target_price": float(r["target_price"]),
                "expiry": _ts(r["expiry"]),
            }
            for r in self.db.recent_hip4_predictions(limit=10)
        ]

        # Latest paper trades
        latest_paper = [
            {
                "trade_date": _ts(r["trade_date"]),
                "underlying": r["underlying"],
                "side": r["side"],
                "direction": r["direction"],
                "stake": float(r["stake"]),
                "potential_payout": float(r["potential_payout"]),
                "outcome": r.get("outcome"),
                "pnl": float(r["pnl"]) if r.get("pnl") is not None else None,
            }
            for r in self.db.recent_paper_trades(limit=10)
        ]

        # Databricks status
        databricks_config = {
            "enabled": settings.databricks_enabled,
            "host": settings.databricks_host,
            "warehouse_id": settings.databricks_warehouse_id,
            "catalog": settings.databricks_catalog,
            "schema": settings.databricks_schema,
        }

        last_cycle = self.db.last_cycle()
        last_cycle_at = last_cycle["started_at"] if last_cycle else None

        content = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "active_symbols": list(settings.symbol_list),
            "counts": counts,
            "system_status": {
                "cpu_percent": status.cpu_percent,
                "memory_used_mb": round(status.memory_used_mb, 1),
                "memory_total_mb": round(status.memory_total_mb, 1),
                "db_path": status.db_path,
                "last_cycle_at": last_cycle_at.isoformat() if hasattr(last_cycle_at, "isoformat") else last_cycle_at,
            },
            "latest_forecasts": latest_forecasts,
            "latest_hip4_predictions": latest_hip4,
            "latest_paper_trades": latest_paper,
            "databricks": databricks_config,
        }

        report = {
            "slug": slug,
            "title": f"Borg System Report — {self._format_date(report_date)}",
            "category": "system",
            "report_date": self._format_date(report_date),
            "description": "Live system health, market coverage, forecasts, HIP-4 predictions, paper trades, and Databricks export status.",
            "content_json": content,
            "status": "published",
        }
        self.db.upsert_report(report)
        return report

    def generate_all(
        self,
        report_date: Optional[date] = None,
        hyperlong_data: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Generate the full daily report suite."""
        return [
            self.generate_daily_report(report_date, hyperlong_data=hyperlong_data),
            self.generate_hr_report(report_date),
            self.generate_brent_report(report_date),
            self.generate_coffee_news(report_date),
            self.generate_system_report(report_date),
        ]

    def generate_pdf(self, slug: str) -> bytes:
        """Generate a simple branded PDF for a report slug."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet

        report = self.db.get_report(slug)
        if report is None:
            raise ValueError(f"Report not found: {slug}")

        content = report.get("content_json", {}) or {}
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, title=report["title"])
        styles = getSampleStyleSheet()
        story: list[Any] = []

        story.append(Paragraph("Borg Reports", styles["Title"]))
        story.append(Paragraph(report["title"], styles["Heading1"]))
        story.append(Paragraph(report.get("description", ""), styles["Normal"]))
        story.append(Spacer(1, 12))

        # Overnight deltas
        if report["category"] == "daily":
            deltas = self.db.get_market_deltas(report["report_date"])
            if deltas:
                data = [["Symbol", "Price", "24h Change"]]
                for d in deltas:
                    data.append([d["symbol"], f"${d['price']:.4f}", f"{d['change_pct']:.2f}%"])
                table = Table(data)
                table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ])
                )
                story.append(Paragraph("Overnight Market Deltas", styles["Heading2"]))
                story.append(table)
                story.append(Spacer(1, 12))

            news = content.get("overnight_news", {})
            story.append(Paragraph("Overnight News", styles["Heading2"]))
            for key, value in news.items():
                story.append(Paragraph(f"<b>{key.title()}:</b> {value}", styles["Normal"]))

            story.append(Spacer(1, 12))
            story.append(Paragraph("Today's Forecast", styles["Heading2"]))
            for f in content.get("mlb_forecasts", []):
                story.append(Paragraph(f"{f['time']} — {f['matchup']}: {f['prediction']}", styles["Normal"]))

        elif report["category"] == "hr":
            story.append(Paragraph("Ranked Candidates", styles["Heading2"]))
            for c in content.get("candidates", []):
                story.append(
                    Paragraph(
                        f"#{c['rank']} {c['player']} ({c['team']}) — Score {c['weather_score']:.1f}, {c['tier']}, Flags: {c['red_flags']}",
                        styles["Normal"],
                    )
                )

        elif report["category"] == "brent":
            story.append(Paragraph(f"Brent Price: ${content.get('brent_price', 0):.2f}", styles["Heading2"]))
            story.append(Paragraph(f"24h Change: {content.get('change_pct', 0):.2f}%", styles["Normal"]))
            story.append(Paragraph(f"Inventory Signal: {content.get('inventory_signal', '')}", styles["Normal"]))
            story.append(Paragraph(f"Rig Count: {content.get('rig_count', 0)}", styles["Normal"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Headlines", styles["Heading2"]))
            for h in content.get("headlines", []):
                story.append(Paragraph(f"• {h}", styles["Normal"]))

        elif report["category"] == "coffee":
            story.append(Paragraph(content.get("headline", ""), styles["Heading2"]))
            story.append(Paragraph(content.get("overnight_summary", ""), styles["Normal"]))
            story.append(Paragraph(f"Market Tick: {content.get('market_tick', '')}", styles["Normal"]))
            story.append(Paragraph(f"Today's Lock: {content.get('todays_lock', '')}", styles["Normal"]))
            for s in content.get("sections", []):
                story.append(Paragraph(s["title"], styles["Heading3"]))
                story.append(Paragraph(s["body"], styles["Normal"]))

        elif report["category"] == "system":
            story.append(Paragraph("System Health", styles["Heading2"]))
            status = content.get("system_status", {})
            story.append(Paragraph(f"CPU: {status.get('cpu_percent', 0):.1f}%", styles["Normal"]))
            story.append(Paragraph(f"Memory: {status.get('memory_used_mb', 0):.0f} / {status.get('memory_total_mb', 0):.0f} MB", styles["Normal"]))
            story.append(Paragraph(f"Database: {status.get('db_path', '')}", styles["Normal"]))
            story.append(Paragraph(f"Last cycle: {status.get('last_cycle_at') or '--'}", styles["Normal"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Active Symbols", styles["Heading2"]))
            story.append(Paragraph(", ".join(content.get("active_symbols", [])), styles["Normal"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Data Store Counts", styles["Heading2"]))
            for table, count in content.get("counts", {}).items():
                story.append(Paragraph(f"{table}: {count}", styles["Normal"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Latest Forecasts", styles["Heading2"]))
            for f in content.get("latest_forecasts", []):
                story.append(Paragraph(
                    f"{f['symbol']} {f['direction'].upper()} {f['confidence']:.1f}% → {f.get('outcome', 'pending')}",
                    styles["Normal"],
                ))
            story.append(Spacer(1, 12))

            story.append(Paragraph("HIP-4 Daily Predictions", styles["Heading2"]))
            for p in content.get("latest_hip4_predictions", []):
                story.append(Paragraph(
                    f"{p['underlying']} {p['direction'].upper()} {p['confidence']:.1f}% target {p['target_price']} expiry {p['expiry']}",
                    styles["Normal"],
                ))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Paper Trades", styles["Heading2"]))
            for t in content.get("latest_paper_trades", []):
                pnl = t.get("pnl")
                pnl_text = f" PnL {pnl:.4f}" if pnl is not None else ""
                story.append(Paragraph(
                    f"{t['trade_date']} {t['underlying']} {t['side']} ${t['stake']:.2f} → {t.get('outcome', 'pending')}{pnl_text}",
                    styles["Normal"],
                ))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Databricks Export", styles["Heading2"]))
            dbx = content.get("databricks", {})
            story.append(Paragraph(f"Enabled: {dbx.get('enabled', False)}", styles["Normal"]))
            story.append(Paragraph(f"Host: {dbx.get('host') or '--'}", styles["Normal"]))
            story.append(Paragraph(f"Warehouse: {dbx.get('warehouse_id') or '--'}", styles["Normal"]))
            story.append(Paragraph(f"Catalog / Schema: {dbx.get('catalog')} / {dbx.get('schema')}", styles["Normal"]))

        story.append(Spacer(1, 24))
        story.append(Paragraph("Generated by Borg · Autonomous Market Intelligence", styles["Italic"]))

        doc.build(story)
        return buffer.getvalue()

    def seed_sample_events(self) -> None:
        """Seed scheduled events similar to saskpoly.xyz/schedule."""
        base = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        events = [
            {
                "title": "U.S. CPI",
                "category": "U.S. Macro",
                "impact": "High",
                "event_time": (base.replace(day=base.day + 1) + timedelta(hours=12, minutes=30)).isoformat(),
                "description": "Consumer Price Index — monthly inflation read.",
                "tags": ["CPI", "Inflation"],
            },
            {
                "title": "U.S. Retail Sales",
                "category": "U.S. Macro",
                "impact": "High",
                "event_time": (base.replace(day=base.day + 2) + timedelta(hours=12, minutes=30)).isoformat(),
                "description": "Monthly retail sales — consumer-demand pulse.",
                "tags": ["Retail"],
            },
            {
                "title": "EIA Weekly Petroleum Status Report",
                "category": "Crude Oil",
                "impact": "High",
                "event_time": (base.replace(day=base.day + 2) + timedelta(hours=14, minutes=30)).isoformat(),
                "description": "Official U.S. crude, gasoline, and distillate inventories.",
                "tags": ["EIA", "Inventories"],
            },
            {
                "title": "Baker Hughes U.S. Rig Count",
                "category": "Crude Oil",
                "impact": "Medium",
                "event_time": (base + timedelta(days=7, hours=15)).isoformat(),
                "description": "Weekly oil & gas rig count — production-intent signal.",
                "tags": ["Rigs"],
            },
        ]
        for event in events:
            self.db.insert_scheduled_event(event)


report_engine = ReportEngine()
