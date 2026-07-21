"""FastAPI web dashboard and API for the Borg prototype."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from borg.brain import brain
from borg.chat import chat_engine
from borg.config import settings
from borg.conscious import consciousness
from borg.db import db
from borg.events import event_log
from borg.ingest import watcher
from borg.llm import llm
from borg.memory import memory
from borg.monitor import monitor
from borg.modules.reports import report_engine
from borg.modules.self_improve import self_improve
from borg.modules.strategy_report import generate_strategy_report
from borg.modules.smb_inventory import inventory
from borg.modules.camera import camera_manager
from borg.modules.databricks_export import export_all
from borg.modules.diary import diary_writer, list_diary_files
from borg.modules.hyperlong_client import fetch_all_symbols, fetch_chart_data
from borg.modules.image_gen import image_client
from borg.reasoning_audit import ReasoningAudit
from borg.schemas import CandleInput, ForecastInput, ImageGenerationInput, SystemStatus
from borg.safety import ActionKind, safety
from borg.versioning import versioning
from borg.visual.sim import colony
from borg.web.auth import AuthMiddleware, auth

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Background task handles started by the lifespan manager.
_colony_ticker_task: asyncio.Task | None = None
_ingest_task: asyncio.Task | None = None
_conscious_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown background tasks for the web app."""
    global _colony_ticker_task, _ingest_task, _conscious_task

    async def _tick_loop() -> None:
        while True:
            colony.tick()
            await asyncio.sleep(0.1)

    async def _ingest_loop() -> None:
        while True:
            await watcher.scan()
            await asyncio.sleep(5)

    async def _conscious_loop() -> None:
        while True:
            try:
                await consciousness.summarize()
            except Exception as exc:
                event_log.emit(f"Consciousness error: {exc}", category="system", phase="idle", level="ERROR")
            await asyncio.sleep(45)

    _colony_ticker_task = asyncio.create_task(_tick_loop())
    _ingest_task = asyncio.create_task(_ingest_loop())
    _conscious_task = asyncio.create_task(_conscious_loop())
    camera_manager.start_all()

    yield

    for task in (_colony_ticker_task, _ingest_task, _conscious_task):
        if task:
            task.cancel()
    camera_manager.stop_all()


app = FastAPI(title="Borg Prototype", version="0.2.0", lifespan=lifespan)
app.add_middleware(AuthMiddleware)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _system_status() -> SystemStatus:
    status = monitor.status()
    status.ollama_reachable = await llm._check_ollama()
    last = db.last_cycle()
    if last:
        started = last["started_at"]
        if isinstance(started, datetime):
            status.last_cycle_at = started
        else:
            try:
                status.last_cycle_at = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            except Exception:
                status.last_cycle_at = None
    return status


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "postgres": settings.is_postgres}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str | None = None) -> HTMLResponse:
    if not auth.is_enabled():
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login")
async def login_post(request: Request, password: str = Form(...)) -> Response:
    token = auth.login(password)
    if token is None:
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid password"}, status_code=401)
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(auth.COOKIE_NAME, token, httponly=True, max_age=86400)
    return response


@app.get("/logout")
async def logout() -> Response:
    response = RedirectResponse(url="/login", status_code=302)
    auth.logout(response)
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    status = await _system_status()
    forecasts = db.recent_forecasts(limit=20)
    learnings = memory.recent(limit=10)
    cycles = db.fetchall("SELECT * FROM brain_cycles ORDER BY started_at DESC LIMIT 10")
    goals = db.list_goals(limit=10)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "active_page": "dashboard",
            "status": status,
            "forecasts": forecasts,
            "learnings": learnings,
            "cycles": cycles,
            "goals": goals,
        },
    )


@app.get("/reports/forecasts", response_class=HTMLResponse)
async def report_forecasts(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/forecasts.html", {"active_page": "forecasts"})


@app.get("/reports/learnings", response_class=HTMLResponse)
async def report_learnings(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/learnings.html", {"active_page": "learnings"})


@app.get("/reports/goals-tasks", response_class=HTMLResponse)
async def report_goals_tasks(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/goals_tasks.html", {"active_page": "goals-tasks"})


@app.get("/reports/events", response_class=HTMLResponse)
async def report_events(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/events.html", {"active_page": "events"})


@app.get("/reports/audit", response_class=HTMLResponse)
async def report_audit(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/audit.html", {"active_page": "audit"})


@app.get("/reports/versions", response_class=HTMLResponse)
async def report_versions(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/versions.html", {"active_page": "versions"})


@app.get("/reports/candles", response_class=HTMLResponse)
async def report_candles(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/candles.html", {"active_page": "candles"})


@app.get("/reports/marketplace", response_class=HTMLResponse)
async def report_marketplace(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/marketplace.html", {"active_page": "marketplace"})


@app.get("/reports/schedule", response_class=HTMLResponse)
async def report_schedule(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "reports/schedule.html", {"active_page": "schedule"})


@app.get("/reports/paper-trades", response_class=HTMLResponse)
async def report_paper_trades(request: Request) -> HTMLResponse:
    """Dashboard showing daily HIP-4 paper-trade history."""
    return templates.TemplateResponse(request, "reports/paper_trades.html", {"active_page": "paper-trades"})


@app.get("/diary", response_class=HTMLResponse)
async def diary_page(request: Request) -> HTMLResponse:
    """Daily diary list and viewer."""
    return templates.TemplateResponse(
        request,
        "diary.html",
        {"active_page": "diary", "output_dir": str(settings.diary_output_path)},
    )


@app.get("/reports/view/{slug}", response_class=HTMLResponse)
async def report_detail(request: Request, slug: str) -> HTMLResponse:
    report = db.get_report(slug)
    if report is None:
        return templates.TemplateResponse(request, "reports/marketplace.html", {"active_page": "marketplace", "error": f"Report {slug} not found"}, status_code=404)

    template_map = {
        "daily": "reports/daily.html",
        "hr": "reports/hr.html",
        "brent": "reports/brent.html",
        "coffee": "reports/coffee.html",
        "system": "reports/system.html",
    }
    template = template_map.get(report["category"], "reports/daily.html")
    deltas = db.get_market_deltas(report["report_date"]) if report["category"] == "daily" else []
    return templates.TemplateResponse(
        request,
        template,
        {
            "active_page": "marketplace",
            "report": report,
            "content": report.get("content_json", {}),
            "deltas": deltas,
        },
    )


@app.get("/api/reports")
async def api_list_reports(category: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return db.list_reports(category=category, limit=limit)


@app.get("/api/reports/{slug}")
async def api_get_report(slug: str) -> dict[str, Any]:
    report = db.get_report(slug)
    if report is None:
        return {"status": "error", "message": f"Report {slug} not found"}
    report["deltas"] = db.get_market_deltas(report["report_date"]) if report["category"] == "daily" else []
    return report


@app.post("/api/reports/generate")
async def api_generate_reports() -> dict[str, Any]:
    hyperlong_data = await fetch_all_symbols()
    generated = report_engine.generate_all(hyperlong_data=hyperlong_data)
    return {"status": "ok", "generated": [r["slug"] for r in generated]}


@app.get("/api/reports/{slug}/pdf")
async def api_report_pdf(slug: str) -> Response:
    try:
        pdf_bytes = report_engine.generate_pdf(slug)
        return Response(
            pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={slug}.pdf"},
        )
    except Exception as exc:
        return Response(f"PDF generation failed: {exc}", media_type="text/plain", status_code=500)


@app.get("/api/schedule")
async def api_schedule(after: str | None = None, before: str | None = None, category: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    return db.list_scheduled_events(after=after, before=before, category=category, limit=limit)


@app.post("/api/schedule/seed")
async def api_seed_schedule() -> dict[str, Any]:
    report_engine.seed_sample_events()
    return {"status": "ok", "message": "Sample events seeded"}


@app.get("/api/status")
async def api_status() -> SystemStatus:
    return await _system_status()


@app.get("/api/strategies")
async def list_strategies() -> dict[str, Any]:
    """List loaded trading strategies and their risk metrics."""
    from borg.coordinator import StrategyCoordinator

    coordinator = StrategyCoordinator(database=db, llm=llm)
    return {
        "strategies": [
            {
                "name": name,
                "enabled": strategy.config.get("enabled", True),
                "risk_metrics": strategy.risk_metrics(),
            }
            for name, strategy in coordinator.strategies.items()
        ]
    }


@app.get("/api/strategies/report")
async def strategy_report() -> dict[str, Any]:
    """Return today's aggregated strategy signal report."""
    return generate_strategy_report()


@app.get("/api/llm/stats")
async def llm_stats() -> dict[str, Any]:
    """Return hybrid LLM usage statistics."""
    from borg.llm_hybrid import hybrid_llm

    return hybrid_llm.stats_summary()


@app.post("/api/candles")
async def create_candle(candle: CandleInput) -> dict[str, Any]:
    row_id = db.insert_candle(candle.model_dump())
    return {"status": "ok", "id": row_id}


@app.post("/api/candles/upload")
async def upload_candles(file: UploadFile) -> dict[str, Any]:
    import csv
    import os
    import tempfile

    inserted = 0
    skipped = 0
    errors: list[tuple[int, str]] = []
    content = await file.read()
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        with open(tmp_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=1):
                try:
                    validated = CandleInput(**row)
                    db.insert_candle(validated.model_dump())
                    inserted += 1
                except Exception as exc:
                    errors.append((row_num, str(exc)))
                    skipped += 1
    finally:
        os.unlink(tmp_path)

    return {"inserted": inserted, "skipped": skipped, "errors": errors[:20]}


@app.get("/api/candles/{symbol}")
async def get_candles(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    return db.latest_candles(symbol.upper(), limit=limit)


@app.post("/api/forecasts")
async def create_forecast(forecast: ForecastInput) -> dict[str, Any]:
    row_id = db.insert_forecast(
        {
            "symbol": forecast.symbol,
            "direction": forecast.direction.value,
            "confidence": forecast.confidence,
            "rationale": forecast.rationale,
            "model_used": "manual",
        }
    )
    return {"status": "ok", "id": row_id}


@app.get("/api/forecasts")
async def list_forecasts(symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return db.recent_forecasts(symbol=symbol, limit=limit)


@app.get("/api/hip4_predictions")
async def list_hip4_predictions(underlying: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Return active HIP-4 daily binary-option predictions."""
    return db.recent_hip4_predictions(underlying=underlying.upper() if underlying else None, limit=limit)


@app.get("/api/paper_trades")
async def list_paper_trades(limit: int = 50) -> list[dict[str, Any]]:
    """Return daily HIP-4 paper-trade history."""
    return db.recent_paper_trades(limit=limit)


@app.get("/api/diary")
async def api_list_diaries() -> list[dict[str, Any]]:
    """List generated daily diary files."""
    return list_diary_files()


@app.post("/api/diary/generate")
async def api_generate_diary() -> dict[str, Any]:
    """Generate today's diary on demand."""
    today = datetime.now(timezone.utc).date()

    try:
        hyperlong_data = await fetch_all_symbols()
        path = diary_writer.write_daily_diary(today, hyperlong_data=hyperlong_data)
        return {"status": "ok", "path": str(path), "date": today.isoformat()}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.get("/api/diary/{date_str}/download")
async def api_download_diary(date_str: str) -> Response:
    """Return a diary Markdown file as a downloadable response."""
    path = diary_writer.output_dir / f"{date_str}.md"
    if not path.exists():
        return Response(f"Diary for {date_str} not found", media_type="text/plain", status_code=404)
    return Response(
        path.read_text(encoding="utf-8"),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={date_str}.md"},
    )


@app.get("/api/diary/{date_str}")
async def api_get_diary(date_str: str) -> dict[str, Any]:
    """Return the Markdown content of a specific diary file."""
    path = diary_writer.output_dir / f"{date_str}.md"
    if not path.exists():
        return {"status": "error", "message": f"Diary for {date_str} not found"}
    return {"status": "ok", "date": date_str, "content": path.read_text(encoding="utf-8")}


@app.get("/api/hyperlong/{symbol}")
async def api_hyperlong_symbol(symbol: str) -> dict[str, Any]:
    """Return HyperLong chart/indicator data for a symbol."""
    return await fetch_chart_data(symbol)


@app.get("/api/hyperlong")
async def api_hyperlong_all() -> dict[str, Any]:
    """Return HyperLong chart/indicator data for all watched symbols."""
    return await fetch_all_symbols()


@app.post("/api/databricks/export")
async def api_databricks_export() -> dict[str, Any]:
    """Push forecasts, HIP-4 predictions, paper trades, and candles to Databricks."""
    return await export_all()


@app.get("/api/databricks/config")
async def api_databricks_config() -> dict[str, Any]:
    """Show whether Databricks export is configured (without revealing the token)."""
    return {
        "enabled": settings.databricks_enabled,
        "host": settings.databricks_host,
        "warehouse_id": settings.databricks_warehouse_id,
        "catalog": settings.databricks_catalog,
        "schema": settings.databricks_schema,
        "tables": settings.databricks_tables,
        "missing": [
            var
            for var, val in {
                "DATABRICKS_HOST": settings.databricks_host,
                "DATABRICKS_TOKEN": settings.databricks_token,
                "DATABRICKS_WAREHOUSE_ID": settings.databricks_warehouse_id,
            }.items()
            if not val
        ],
    }


@app.get("/api/forecasts/{forecast_id}/reasoning")
async def get_forecast_reasoning(forecast_id: int) -> dict[str, Any]:
    """Return the full reasoning chain for a forecast."""
    rows = db.fetchall(
        "SELECT * FROM forecasts WHERE id = %s" if db.is_postgres else "SELECT * FROM forecasts WHERE id = ?",
        (forecast_id,),
    )
    if not rows:
        return {"error": "Forecast not found"}
    forecast = db._row_to_dict(rows[0])
    forecast["features"] = db._from_json(forecast.get("features"))
    forecast["reasoning_output"] = db._from_json(forecast.get("reasoning_output"))
    return {
        "forecast_id": forecast_id,
        "symbol": forecast["symbol"],
        "direction": forecast["direction"],
        "confidence": forecast["confidence"],
        "reasoning_output": forecast["reasoning_output"],
        "outcome": forecast["outcome"],
        "correct": forecast["correct"],
    }


@app.get("/api/reasoning/calibration")
async def get_reasoning_calibration(window_days: int = 30) -> dict[str, Any]:
    """Return reasoning calibration report."""
    audit = ReasoningAudit(database=db)
    return await audit.get_calibration_report(window_days=window_days)


@app.get("/api/learning/history")
async def get_learning_history(limit: int = 30) -> list[dict[str, Any]]:
    """Return history of autonomous learning updates."""
    audits = db.recent_audit(limit=limit)
    return [
        {
            "timestamp": a["created_at"],
            "actor": a["actor"],
            "action": a["action"],
            "detail": a.get("detail", {}),
        }
        for a in audits
        if a.get("action") == "learning_update"
    ]


@app.get("/dashboard/learning-performance")
async def learning_dashboard(request: Request) -> HTMLResponse:
    """Dashboard showing learning impact over time."""
    audits = [
        a
        for a in db.recent_audit(limit=50)
        if a.get("action") == "learning_update"
    ]
    return templates.TemplateResponse(
        request,
        "learning_dashboard.html",
        {
            "active_page": "learning-performance",
            "updates": audits,
        },
    )


@app.get("/api/consciousness/daily")
async def get_daily_consciousness_report() -> dict[str, Any]:
    """Return today's consciousness report, generating it if needed."""
    today = datetime.now(timezone.utc).date().isoformat()
    report = db.get_consciousness_report("daily", today)
    if report:
        return {"report": report["report_text"], "generated": False, "score": report.get("score")}
    return {"report": "No report generated yet", "generated": False, "score": None}


@app.get("/api/consciousness/weekly")
async def get_weekly_consciousness_report() -> dict[str, Any]:
    """Return this week's consciousness report if available."""
    today = datetime.now(timezone.utc).date().isoformat()
    report = db.get_consciousness_report("weekly", today)
    if report:
        return {"report": report["report_text"], "generated": False}
    return {"report": "No weekly report generated yet", "generated": False}


@app.get("/api/consciousness/score")
async def get_consciousness_score() -> dict[str, Any]:
    """Return the latest consciousness score."""
    from borg.consciousness_score import ConsciousnessScore

    recent_reports = db.query_consciousness_reports(limit=1000)
    recent_audits = db.recent_audit(limit=100)
    learning_updates = [a for a in recent_audits if a.get("action") == "learning_update"]
    score = ConsciousnessScore.calculate(
        reasoning_accuracy=0.75,
        learning_updates_count=len(learning_updates),
        report_count=len(recent_reports),
        performance_consistency=0.6,
    )
    return {"score": score, "report_count": len(recent_reports), "learning_updates": len(learning_updates)}


@app.get("/dashboard/consciousness")
async def consciousness_dashboard(request: Request) -> HTMLResponse:
    """Dashboard showing consciousness reports and score."""
    today = datetime.now(timezone.utc).date().isoformat()
    daily = db.get_consciousness_report("daily", today)
    weekly = db.get_consciousness_report("weekly", today)
    recent_reports = db.query_consciousness_reports(limit=7)
    return templates.TemplateResponse(
        request,
        "consciousness_dashboard.html",
        {
            "active_page": "consciousness",
            "daily_report": daily["report_text"] if daily else "Generating...",
            "weekly_report": weekly["report_text"] if weekly else "Generating...",
            "recent_reports": recent_reports,
        },
    )


@app.get("/dashboard/reasoning-audit")
async def reasoning_audit_dashboard(request: Request) -> HTMLResponse:
    """Dashboard showing reasoning chains and calibration."""
    audit = ReasoningAudit(database=db)
    calibration = await audit.get_calibration_report(window_days=30)
    recent = db.query_reasoning_audits(limit=50)
    return templates.TemplateResponse(
        request,
        "reasoning_audit.html",
        {
            "active_page": "reasoning-audit",
            "calibration": calibration,
            "recent_audits": recent,
        },
    )


@app.get("/api/learnings")
async def query_learnings(q: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    if q:
        return await memory.recall(q, top_k=limit)
    return memory.recent(limit=limit)


@app.post("/api/learnings")
async def create_learning(summary: str = Form(...), detail: str = Form(""), tags: str = Form("")) -> dict[str, Any]:
    row_id = await memory.remember(summary=summary, detail=detail, tags=tags.split(",") if tags else None)
    return {"status": "ok", "id": row_id}


@app.post("/api/brain/step")
async def brain_step() -> dict[str, Any]:
    return await brain.cycle()


@app.post("/api/brain/seed")
async def brain_seed() -> dict[str, Any]:
    await brain.seed()
    return {"status": "ok", "message": "Candle history seeded (real + synthetic fallback)"}


@app.post("/api/analyze")
async def analyze(symbol: str = Form(...), summary: str = Form(...)) -> dict[str, Any]:
    return await llm.analyze_market(symbol.upper(), summary)


@app.get("/api/events")
async def list_events(limit: int = 50, after_id: int = 0, category: str | None = None) -> list[dict[str, Any]]:
    return db.recent_events(limit=limit, after_id=after_id, category=category)


@app.post("/api/events")
async def create_event(message: str = Form(...), category: str = Form("user"), level: str = Form("INFO")) -> dict[str, Any]:
    row_id = event_log.emit(message, category=category, level=level)
    return {"status": "ok", "id": row_id}


@app.get("/api/visual/state")
async def visual_state() -> dict[str, Any]:
    return colony.state()


@app.get("/api/goals")
async def list_goals(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return db.list_goals(status=status, limit=limit)


@app.post("/api/goals")
async def create_goal(title: str = Form(...), description: str = Form(""), priority: int = Form(50)) -> dict[str, Any]:
    goal_id = db.create_goal(title, description, priority)
    return {"status": "ok", "id": goal_id}


@app.post("/api/goals/{goal_id}/status")
async def update_goal_status(goal_id: int, status: str = Form(...)) -> dict[str, Any]:
    db.update_goal_status(goal_id, status)
    return {"status": "ok"}


@app.get("/api/tasks")
async def list_tasks(status: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return db.list_tasks(status=status, kind=kind, limit=limit)


@app.get("/api/audit")
async def list_audit(limit: int = 50) -> list[dict[str, Any]]:
    return db.recent_audit(limit=limit)


@app.get("/api/versions")
async def list_versions(module: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return db.list_versions(module=module, limit=limit)


@app.post("/api/versions/{version_id}/apply")
async def apply_version(version_id: int) -> dict[str, Any]:
    return versioning.apply(version_id)


@app.post("/api/versions/{version_id}/reject")
async def reject_version(version_id: int, reason: str = Form("")) -> dict[str, Any]:
    return versioning.reject(version_id, reason=reason)


@app.post("/api/self-improve")
async def trigger_self_improve() -> dict[str, Any]:
    proposal = await self_improve.analyze()
    return proposal or {"status": "ok", "message": "No improvement proposed (not enough learnings yet)"}


@app.post("/api/ingest")
async def trigger_ingest() -> dict[str, Any]:
    results = await watcher.scan()
    return {"status": "ok", "processed": len(results), "results": results}


@app.post("/api/chat")
async def chat(message: str = Form(...)) -> dict[str, Any]:
    return await chat_engine.ask(message)


@app.get("/api/chat/history")
async def chat_history(limit: int = 50) -> list[dict[str, Any]]:
    return chat_engine.history(limit=limit)


# Image generation -----------------------------------------------------------


@app.get("/image-gen", response_class=HTMLResponse)
async def image_gen_page(request: Request) -> HTMLResponse:
    models = await image_client.list_models()
    return templates.TemplateResponse(
        request,
        "image_gen.html",
        {
            "active_page": "image-gen",
            "models": models,
            "needs_confirmation": safety.needs_confirmation(ActionKind.IMAGE_GENERATION),
        },
    )


@app.get("/api/images/models")
async def api_image_models() -> list[dict[str, Any]]:
    return await image_client.list_models()


@app.post("/api/images/generate")
async def api_generate_image(request: ImageGenerationInput) -> dict[str, Any]:
    result = await image_client.generate_from_schema(request)
    return result.model_dump()


# Safety approvals -----------------------------------------------------------


@app.post("/api/safety/{action}/approve")
async def api_approve_action(action: str) -> dict[str, Any]:
    return safety.approve(action)


@app.post("/api/safety/{action}/reject")
async def api_reject_action(action: str, reason: str = Form("")) -> dict[str, Any]:
    return safety.reject(action, reason=reason)


# Camera ----------------------------------------------------------------------


@app.get("/camera", response_class=HTMLResponse)
async def camera_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "camera.html",
        {
            "active_page": "camera",
            "cameras": camera_manager.status(),
            "selected": None,
        },
    )


@app.get("/camera/{name}", response_class=HTMLResponse)
async def camera_page(request: Request, name: str) -> HTMLResponse:
    status = camera_manager.status(name)
    if "error" in status:
        return templates.TemplateResponse(
            request,
            "camera.html",
            {"active_page": "camera", "cameras": camera_manager.status(), "selected": None, "error": status["error"]},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "camera.html",
        {
            "active_page": "camera",
            "cameras": camera_manager.status(),
            "selected": status,
        },
    )


@app.get("/api/camera")
async def api_camera_list() -> list[dict[str, Any]]:
    return camera_manager.status()  # type: ignore[return-value]


@app.get("/api/camera/{name}/status")
async def api_camera_status(name: str) -> dict[str, Any]:
    return camera_manager.status(name)


@app.get("/api/camera/{name}/snapshot")
async def api_camera_snapshot(name: str) -> Response:
    frame = camera_manager.snapshot(name)
    if frame is None:
        return Response("No camera frame available", status_code=503)
    return Response(frame, media_type="image/jpeg")


@app.get("/api/camera/{name}/stream")
async def api_camera_stream(name: str) -> Response:
    client = camera_manager.get_client(name)
    if client is None:
        return Response(f"Camera '{name}' not found", status_code=404)
    if not client.is_available:
        return Response("Camera streaming unavailable (OpenCV not installed)", status_code=503)
    return StreamingResponse(
        camera_manager.stream(name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Inventory / assimilation ---------------------------------------------------


@app.get("/inventory", response_class=HTMLResponse)
async def inventory_page(
    request: Request,
    source: str | None = None,
    status: str | None = None,
) -> HTMLResponse:
    entries = db.get_inventory_entries(source=source, status=status, limit=200)
    return templates.TemplateResponse(
        request,
        "inventory/list.html",
        {
            "active_page": "inventory",
            "entries": entries,
            "filter_source": source,
            "filter_status": status,
        },
    )


@app.get("/inventory/{entry_id}", response_class=HTMLResponse)
async def inventory_detail(request: Request, entry_id: int) -> HTMLResponse:
    entry = db.get_inventory_entry(entry_id)
    if entry is None:
        return templates.TemplateResponse(
            request,
            "inventory/detail.html",
            {"active_page": "inventory", "error": f"Entry {entry_id} not found"},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "inventory/detail.html",
        {"active_page": "inventory", "entry": entry},
    )


@app.post("/inventory/scan")
async def inventory_scan(
    host: str = Form(""),
    share: str = Form(""),
    username: str = Form(""),
    password: str = Form(""),
    root_path: str = Form(""),
) -> RedirectResponse:
    await inventory.scan(
        host=host or None,
        share=share or None,
        username=username or None,
        password=password or None,
        root_path=root_path or None,
    )
    return RedirectResponse(url="/inventory", status_code=302)


@app.post("/inventory/score")
async def inventory_score() -> RedirectResponse:
    await inventory.score_candidates()
    return RedirectResponse(url="/inventory", status_code=302)


@app.post("/inventory/{entry_id}/stage")
async def inventory_stage(entry_id: int) -> RedirectResponse:
    await inventory.stage_candidate(entry_id)
    return RedirectResponse(url=f"/inventory/{entry_id}", status_code=302)


@app.post("/inventory/{entry_id}/approve")
async def inventory_approve(entry_id: int) -> RedirectResponse:
    await inventory.apply_candidate(entry_id)
    return RedirectResponse(url=f"/inventory/{entry_id}", status_code=302)


@app.post("/inventory/{entry_id}/reject")
async def inventory_reject(entry_id: int, reason: str = Form("")) -> RedirectResponse:
    await inventory.reject_candidate(entry_id, reason=reason)
    return RedirectResponse(url=f"/inventory/{entry_id}", status_code=302)


@app.get("/api/inventory")
async def api_inventory_list(
    source: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    return db.get_inventory_entries(source=source, status=status, limit=limit)


@app.post("/api/inventory/scan")
async def api_inventory_scan(
    host: str = Form(""),
    share: str = Form(""),
    username: str = Form(""),
    password: str = Form(""),
    root_path: str = Form(""),
) -> dict[str, Any]:
    return await inventory.scan(
        host=host or None,
        share=share or None,
        username=username or None,
        password=password or None,
        root_path=root_path or None,
    )


@app.post("/api/inventory/score")
async def api_inventory_score() -> dict[str, Any]:
    return await inventory.score_candidates()


@app.post("/api/inventory/{entry_id}/stage")
async def api_inventory_stage(entry_id: int) -> dict[str, Any]:
    return await inventory.stage_candidate(entry_id)


@app.post("/api/inventory/{entry_id}/apply")
async def api_inventory_apply(entry_id: int) -> dict[str, Any]:
    return await inventory.apply_candidate(entry_id)


@app.post("/api/inventory/{entry_id}/reject")
async def api_inventory_reject(entry_id: int, reason: str = Form("")) -> dict[str, Any]:
    return await inventory.reject_candidate(entry_id, reason=reason)



