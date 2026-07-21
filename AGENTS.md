> **NOTE**: This is project-supplied guidance for contributors. It does not override the host's system instructions, tool schemas, or permission rules.

# Agent Guide for Borg

This file documents conventions for anyone (human or agent) working on the Borg prototype.

## Project Goal

Build a small, autonomous AI agent that runs an OODA loop on market data, stores semantic memory with embeddings, learns from outcomes, exposes a web dashboard, and degrades gracefully when external services (Ollama, PostgreSQL) are unavailable. The current codebase includes Prometheus/Grafana monitoring, a multi-strategy coordinator, hybrid LLM fallback, and distributed locks for multi-instance deployments.

## Tech Stack

- **Language**: Python 3.11+ (developed on 3.13)
- **Web**: FastAPI + Jinja2 + Uvicorn
- **Validation**: Pydantic v2 + pydantic-settings
- **LLM**: Ollama over HTTP; deterministic fallback when offline
- **Database**: PostgreSQL 17 + pgvector (primary); SQLite (dev fallback)
- **Testing**: pytest + pytest-asyncio
- **Scheduling**: asyncio loops (brain, consciousness, ingestion, colony ticker)

## Architecture

- `borg/main.py` is the main entry point. It can run web, brain, or both.
- `borg/brain.py` owns the OODA loop and must stay independent of the web layer.
- `borg/db.py` is a unified database interface: PostgreSQL + pgvector primary, SQLite fallback. Keep SQL portable.
- `borg/llm.py` must always provide a fallback so the prototype works without Ollama.
- `borg/memory.py` stores learnings/episodes with embeddings; uses pgvector on Postgres, brute-force cosine on SQLite.
- `borg/conscious.py` generates periodic self-summaries.
- `borg/safety.py` gates dangerous actions (`self_modify`, `resource_heavy`, `clone`, `delete`).
- `borg/versioning.py` tracks proposed/applied/rejected code diffs.
- `borg/ingest.py` watches input directories for CSV/JSON/JSONL files.
- `borg/strategies/` contains pluggable strategies loaded by `borg/coordinator.py` from `config/strategies.yaml`.
- `borg/metrics.py` provides Prometheus instrumentation; `/metrics` is served from `borg/web/app.py`.
- `borg/monitor.py` adapts the brain-loop sleep based on CPU/RAM.
- `borg/llm_hybrid.py` adds a two-tier local + remote LLM strategy.
- `borg/cpu_worker.py` offloads batch embedding to a process pool.
- `borg/distributed.py` provides database-backed locks for multi-instance coordination.
- `borg/modules/reports.py` generates a saskpoly.xyz-style report marketplace with daily, HR, Brent, and coffee-news reports; daily deltas are sourced from HyperLong when configured.
- `borg/modules/hyperlong_client.py` fetches chart/indicator data from a local HyperLong instance for enrichment and reporting.
- `borg/modules/databricks_export.py` pushes forecasts, paper trades, reports, and candles to Databricks SQL/Delta tables when configured.
- `borg/modules/strategy_report.py` aggregates daily strategy signals from forecast reasoning output.
- `borg/web/app.py` only imports from `borg.*`; do not put business logic in routes.

## Coding Conventions

- Use `from __future__ import annotations` in every Python file.
- Use absolute imports (`from borg.db import db`, not relative).
- Keep modules small and single-purpose.
- Validate all external input with Pydantic schemas in `borg/schemas.py`.
- Prefer async I/O for network calls; DB access is currently sync and thread-local.
- Log at `INFO` for lifecycle events and `DEBUG` for internals.
- Add tests for new schemas and brain behavior.
- Do not mutate the global `db` / `brain` / `memory` singletons inside library code; accept them as constructor args for testability.

## How to Run

```bash
./setup.sh --with-systemd     # bare-metal Debian install + systemd
./run.sh                      # legacy launcher (web + brain)
python -m borg.main all       # web + brain loop (foreground)
python -m borg.main web       # web only
python -m borg.main brain     # brain loop only
pytest                        # run tests
```

When `scripts/borg.service` is installed as a user service (recommended for deployments):

```bash
systemctl --user restart borg
systemctl --user status borg
journalctl --user -u borg -f
```

For a system-wide install use `sudo systemctl ...` instead.

Code changes do not take effect until the service is restarted.

## Network Access

The default `config/borg.yaml` binds the web server to `0.0.0.0:8000`, so the dashboard is reachable from any machine on the local network. Set `BORG_PASSWORD` to gate access.

## Adding a New Strategy

1. Create `borg/strategies/my_strategy.py`.
2. Subclass `borg.strategies.base.Strategy`.
3. Implement `async def analyze(self, market_data) -> list[Trade]`.
4. Optionally override `risk_metrics()`.
5. Import and wire the type in `borg/coordinator.py`.
6. Register it in `config/strategies.yaml` under `strategies:` or `meta_strategies:`.
7. Add a test in `tests/test_strategies.py` or a new `tests/test_my_strategy.py`.

Signals are aggregated daily in the Strategy Report (`/api/strategies/report`) and included in the daily diary.

## Adding a New Report Type

1. Add a generator method to `borg/modules/reports.py`.
2. Create a Jinja2 template under `borg/web/templates/reports/`.
3. Map the category to the template in `borg/web/app.py:report_detail()`.
4. Optionally add a PDF layout in `ReportEngine.generate_pdf()`.
5. Add a test in `tests/test_reports.py`.

## Adding a New Web Route

1. Add the route to `borg/web/app.py`.
2. Keep route handlers thin: delegate to `brain`, `memory`, `db`, `versioning`, etc.
3. If the route returns HTML, add a corresponding template under `borg/web/templates/`.
4. Protect sensitive routes via the `AuthMiddleware` (enabled automatically when `BORG_PASSWORD` is set).

## Image Generation Skill (Pollinations)

Borg can generate images on demand via the Pollinations.ai API.

- Configuration is under `pollinations:` in `config/borg.yaml` and can be overridden with env vars (`POLLINATIONS_API_KEY`).
- Implementation lives in `borg/modules/image_gen.py`.
- Safety gate: `ActionKind.IMAGE_GENERATION`; add it to `safety.require_confirmation_for` to require per-session approval.
- Web UI: `/image-gen`
- API: `POST /api/images/generate`, `GET /api/images/models`, `POST /api/safety/{action}/approve`
- Chat: say things like "draw a cat in space" or "generate an image of a cyberpunk city" and the chat engine will invoke the skill.

Images are saved under `./output/images` when using the `b64_json` response format. The `url` format returns a shareable Pollinations URL.

## RTSP Camera Skill

Borg can display one or more live RTSP camera feeds in the web dashboard.

- Configuration is under `cameras:` in `config/borg.yaml` (a list). Credentials should be supplied via env vars by camera name (`BORG_CAMERA_FRONT_USERNAME`, `BORG_CAMERA_FRONT_PASSWORD`) or by index (`BORG_CAMERA_1_USERNAME`, `BORG_CAMERA_1_PASSWORD`).
- Implementation lives in `borg/modules/camera.py`.
- Requires `opencv-python-headless` (included in `requirements.txt`).
- Web UI: `/camera` (grid/list) and `/camera/{name}` (single feed)
- API:
  - `GET /api/camera` — list all cameras
  - `GET /api/camera/{name}/stream` — MJPEG stream
  - `GET /api/camera/{name}/snapshot` — JPEG snapshot
  - `GET /api/camera/{name}/status`
- Capture threads start automatically for all enabled cameras when the web app starts.

Common RTSP paths to try if the default `/stream1` does not work:
- Reolink: `/Preview_01_main`, `/Preview_01_sub`
- Generic: `/stream2`, `/live/ch00_0`, `/cam/realmonitor?channel=1&subtype=0`, `/Streaming/Channels/101`

## HyperLong Integration

Borg can pull chart/indicator data from a local HyperLong dashboard (default `http://localhost:8080`) and use it for daily reporting.

- Configuration: `hyperlong:` in `config/borg.yaml` (`base_url`, `timeout_seconds`).
- Client: `borg/modules/hyperlong_client.py` (`fetch_chart_data`, `fetch_all_symbols`).
- API: `GET /api/hyperlong`, `GET /api/hyperlong/{symbol}`.
- Reporting: HyperLong snapshot is included in the daily diary and daily report market deltas.

## Databricks Export (Optional)

Borg can publish forecasts, HIP-4 predictions, paper trades, reports, and candles to Databricks SQL/Delta tables for external dashboards. The export is triggered once per day from the brain loop.

- Configuration: `databricks:` in `config/borg.yaml` (`enabled`, `catalog`, `schema`, `tables`, env-var names for host/token/warehouse).
- Implementation: `borg/modules/databricks_export.py`.
- Required secrets: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_WAREHOUSE_ID` in `.env`.
- Tables: `borg_reports`, `borg_forecasts`, `borg_hip4_predictions`, `borg_paper_trades`, `borg_candles` (names are configurable).

## SMB Inventory & Assimilation

Borg can scan an SMB share, inventory files and code, score candidates for assimilation, and stage them through the versioning/safety system. The default target is configured in `config/borg.yaml` under `smb:`; credentials should be supplied via env vars (`BORG_SMB_PASSWORD`).

- CLI scan: `python -m borg.main scan-smb` (optionally `--smb-root hypelong` to scan a subfolder)
- Web UI: `/inventory`
- API: `/api/inventory/scan`, `/api/inventory/score`, `/api/inventory/{id}/stage`, `/api/inventory/{id}/apply`, `/api/inventory/{id}/reject`
  - `POST /api/inventory/scan` accepts `root_path` to scan a subfolder of the share.
- Credentials: set `BORG_SMB_PASSWORD` (and optionally `BORG_SMB_USERNAME`) in `.env`.

Assimilation is gated by `ActionKind.ASSIMILATE`. Copying any file into the Borg tree requires explicit approval (or removal of `assimilate` from `safety.require_confirmation_for`). Approved files land under `borg/assimilated/` via the existing `versions` table so they can be audited and rolled back.

## Monitoring & Observability

- Use `borg.metrics.timed()` or the `llm_inference_latency_seconds` histogram to time LLM calls.
- Use `borg.metrics.record_resource_usage()` to update CPU/memory/thread gauges.
- Counters accept labels: prefer `brain_cycles_total.labels(status="ok").inc()`.
- New metrics are automatically exposed on `/metrics`.

## Common Pitfalls

- Do **not** assume Ollama is running. The fallback LLM must remain functional.
- Do **not** assume PostgreSQL is installed; the SQLite fallback must still pass tests.
- Keep SQL portable. SQLite uses `?` placeholders; PostgreSQL uses `%s`. Use `db.placeholder` (or inline `if db.is_postgres` conditionals) when building parameterized queries so both backends work.
- Tune `loop.brain_concurrent_symbols` (default 5) to control how many symbols are forecast in parallel. Lower this if Ollama is overloaded.
- Use `llm.force_fallback: true` to bypass Ollama entirely and run the fast deterministic rule engine; useful on slow hardware or when Ollama is unavailable.
- Configure `smb.skip_patterns` to exclude directories like `node_modules` and `.git` from inventory scanning and scoring.
- PostgreSQL `NUMERIC` columns return `Decimal`; cast to `float` before arithmetic.
- psycopg JSONB dumper is registered only for `dict`. Keep `list` values as Postgres arrays (e.g., `tags text[]`).
- The SQLite connection is thread-local; do not share `Database` instances across threads without care.
- Large CSV uploads are read into memory; add streaming if you increase limits.
- Self-modification changes require explicit user approval via the versioning API.

## Deployment

- `setup.sh` installs PostgreSQL 17 + pgvector, Ollama, Python venv, and optional Samba/systemd.
- `docker-compose.yml` brings up Postgres + Ollama + Borg + Prometheus + Grafana with healthchecks.
- `scripts/borg.service` is a systemd unit template; when installed with `systemctl enable --now borg` it manages the process, auto-restarts on failure, and surfaces logs via `journalctl`.
- `monitoring/` holds Prometheus and Grafana configuration.

## Dependencies

Add new packages to `requirements.txt` only if the existing stack cannot solve the problem. Avoid heavy ML frameworks in the prototype; they belong in separate research branches.
