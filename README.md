# Borg Prototype — User & Operator Manual

<p align="center">
  <img src="assets/borg-banner.svg" alt="Borg — Autonomous Market Analysis Agent" width="100%">
</p>

<p align="center">
  <a href="https://github.com/vote4arealclown/borgdeploy/stargazers"><img src="https://img.shields.io/github/stars/vote4arealclown/borgdeploy?style=social" alt="GitHub stars"></a>
</p>

> ⭐ **If you like the concept, please star the repo.** Stars help others find the project and keep development moving.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white" alt="Pydantic">
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/systemd-FFFFFF?logo=systemd&logoColor=black" alt="systemd">
  <img src="https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white" alt="Prometheus">
  <img src="https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white" alt="Grafana">
  <img src="https://img.shields.io/badge/Databricks-FF3621?logo=databricks&logoColor=white" alt="Databricks">
  <img src="https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/HyperLiquid-4A90E2?logo=ethereum&logoColor=white" alt="HyperLiquid">
  <img src="https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white" alt="pytest">
</p>

## Executive Summary

Borg is an autonomous trading research assistant. You point it at a few markets (BTC, ETH, SOL, HYPE by default), and it watches prices, runs eleven algorithmic strategies, scores the signals, and writes a daily Markdown report — all on its own. It keeps a memory of what worked and what didn't, can propose its own code improvements, and exposes everything through a web dashboard. It is designed to run on a small Linux box at home, work offline using a built-in rule engine, and optionally connect to Ollama, Databricks, or a local HyperLong dashboard when you want more horsepower.

A self-contained, autonomous AI agent that runs an **OODA loop** (Observe → Orient/Plan → Act → Reflect) on market data. It forecasts short-term direction, stores semantic memory with embedding-based recall, learns from outcomes, and exposes a web dashboard with a colony-style visualization and a grounded chat interface.

Designed to fit on modest hardware (4 GB RAM Debian) and works **without Ollama** by falling back to a deterministic rule engine. If Ollama is running with `tinyllama:latest` and `nomic-embed-text:latest`, Borg uses the local LLM automatically.

---

## Quick Links

| What | Local | On your network |
|------|-------|-----------------|
| **Dashboard** | http://localhost:8000 | `http://<host-ip>:8000` |
| **Forecasts report** | http://localhost:8000/reports/forecasts | `http://<host-ip>:8000/reports/forecasts` |
| **Learnings report** | http://localhost:8000/reports/learnings | `http://<host-ip>:8000/reports/learnings` |
| **Goals & tasks report** | http://localhost:8000/reports/goals-tasks | `http://<host-ip>:8000/reports/goals-tasks` |
| **Event log report** | http://localhost:8000/reports/events | `http://<host-ip>:8000/reports/events` |
| **Audit report** | http://localhost:8000/reports/audit | `http://<host-ip>:8000/reports/audit` |
| **Versions report** | http://localhost:8000/reports/versions | `http://<host-ip>:8000/reports/versions` |
| **Candles report** | http://localhost:8000/reports/candles | `http://<host-ip>:8000/reports/candles` |
| **Report marketplace** | http://localhost:8000/reports/marketplace | `http://<host-ip>:8000/reports/marketplace` |
| **Event schedule** | http://localhost:8000/reports/schedule | `http://<host-ip>:8000/reports/schedule` |
| **Paper trades** | http://localhost:8000/reports/paper-trades | `http://<host-ip>:8000/reports/paper-trades` |
| **Daily diary** | http://localhost:8000/diary | `http://<host-ip>:8000/diary` |
| **Health check** | http://localhost:8000/healthz | `http://<host-ip>:8000/healthz` |
| **API status** | http://localhost:8000/api/status | `http://<host-ip>:8000/api/status` |
| **Strategies API** | http://localhost:8000/api/strategies | `http://<host-ip>:8000/api/strategies` |
| **Strategy report API** | http://localhost:8000/api/strategies/report | `http://<host-ip>:8000/api/strategies/report` |
| **HyperLong data API** | http://localhost:8000/api/hyperlong | `http://<host-ip>:8000/api/hyperlong` |
| **Inventory API** | http://localhost:8000/api/inventory | `http://<host-ip>:8000/api/inventory` |
| **LLM stats API** | http://localhost:8000/api/llm/stats | `http://<host-ip>:8000/api/llm/stats` |
| **Prometheus metrics** | http://localhost:8000/metrics | `http://<host-ip>:8000/metrics` |
| **Prometheus UI** | http://localhost:9090 | `http://<host-ip>:9090` |
| **Grafana dashboard** | http://localhost:3000 | `http://<host-ip>:3000` |
| **Agent guide** | [`AGENTS.md`](./AGENTS.md) | — |
| **Improvement roadmap** | [`borg_improvements_essay.md`](./borg_improvements_essay.md) | — |
| **Design PDF** | [`borg_hardened_design_v2.pdf`](./borg_hardened_design_v2.pdf) | — |

Replace `<host-ip>` with the machine's LAN IP.

---

## Features

- **Brain loop**: async OODA cycles over configurable symbols; resolves old forecasts and writes learnings.
- **Goals & tasks**: persistent planning layer with active goals and task audit trail.
- **Forecasting**: up/down/flat predictions with confidence, features, and outcome tracking.
- **Reflection**: per-cycle critique and scoring stored in `reflections`.
- **Semantic memory**: learnings + episodic memory with vector search via **pgvector** (PostgreSQL) or brute-force cosine (SQLite fallback).
- **Consciousness thread**: 30–60 second self-summaries generated from recent events and memory.
- **Ingestion**: watches `./input` and `/borg/input` for CSV/JSON/JSONL candle files.
- **Safety gates**: configurable confirmation requirements for `self_modify`, `resource_heavy`, `clone`, `delete`.
- **Versioning**: proposed code diffs, apply/reject workflow, full audit log.
- **Self-improvement**: proposes small code changes based on recent learnings (requires approval).
- **Web dashboard**: FastAPI + Jinja2 UI with live colony map, event log, forecasts, goals, chat, dedicated report pages, and a saskpoly.xyz-style report marketplace.
- **Daily Diary**: end-of-day Markdown summaries of forecasts, HIP-4 paper trades, learnings, reflections, and events. View at `/diary` or download via `/api/diary/{date}/download`.
- **Chat assistant**: answers from live Borg data (status, forecasts, learnings, events, day/time).
- **Auth**: optional password gate via `BORG_PASSWORD`.
- **System monitor**: adaptive loop sleep based on CPU/RAM; throttles under load.
- **Pluggable strategies**: 11 algorithmic strategies including binary forecast, mean reversion, momentum, pairs trading, trend following, statistical arbitrage, VWAP, TWAP, grid trading, market making, and DCA. Loaded from `config/strategies.yaml`.
- **Strategy reporting**: daily aggregated signal counts per symbol and per strategy, available via `/api/strategies/report` and included in the daily diary.
- **HyperLong integration**: fetches chart/indicator data from a local HyperLong dashboard (default `http://localhost:8080`) and includes a snapshot in the daily diary and daily report.
- **Databricks export**: optionally publishes forecasts, paper trades, reports, and candles to Databricks SQL/Delta tables for external dashboards.
- **Hybrid LLM**: fast local Ollama first; escalates low-confidence forecasts to OpenAI-compatible models when configured.
- **Prometheus + Grafana**: `/metrics` endpoint, compose services, and a pre-built Borg System Overview dashboard.
- **Profiling**: `scripts/profile_brain.py` breaks down cycle latency by phase.
- **Distributed locks**: database-backed locks for safe multi-instance coordination.
- **Deployment**: `setup.sh` for bare-metal Debian, `docker-compose.yml` for containers, `scripts/borg.service` for systemd (installed as a user service by default).

---

## Quick Start

### Bare metal (Debian/Ubuntu)

```bash
# Full install: Borg + monitoring (Prometheus/Grafana) + systemd auto-start
./setup.sh --with-systemd --with-monitoring
```

Then open `http://<host-ip>:8000` from any machine on the network. Prometheus is on port `9090` and Grafana on port `3000`.

### Docker

```bash
docker compose up -d --build
docker compose exec ollama ollama pull tinyllama:latest
docker compose exec ollama ollama pull nomic-embed-text:latest
```

### Manual start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and edit env
# cp .env.example .env

# Web + brain loop together (listens on all interfaces by default)
python -m borg.main all

# Or run just the web API
python -m borg.main web

# Or run just the brain loop
python -m borg.main brain
```

When `scripts/borg.service` is installed as a user service, use `systemctl --user` instead:

```bash
systemctl --user restart borg
systemctl --user status borg
journalctl --user -u borg -f
```

For a system-wide service, use `sudo systemctl ...` instead.

---

## Project Layout

```
borg/
  main.py                # Boot point: web + brain + background tasks
  config.py              # YAML + env settings via pydantic-settings
  db.py                  # PostgreSQL + pgvector primary, SQLite fallback
  schemas.py             # Pydantic input/output models
  llm.py                 # Ollama client + OpenAI-compatible fallback + rule engine
  llm_hybrid.py          # Two-tier local + remote LLM strategy
  brain.py               # OODA loop, synthetic data feed, forecast resolution
  memory.py              # Embedding-based memory / learnings
  monitor.py             # CPU / memory telemetry and adaptive throttling
  metrics.py             # Prometheus instrumentation
  events.py              # Structured event log
  conscious.py           # Periodic self-reflection thread
  safety.py              # Confirmation gates
  versioning.py          # Proposed code changes with approval workflow
  ingest.py              # CSV/JSON/JSONL file watcher
  cpu_worker.py          # Process-pool embedding worker
  distributed.py         # Database-backed distributed locks
  coordinator.py         # Strategy loader / orchestrator
  modules/
    binary_options.py    # Technical-analysis helpers
    self_improve.py      # Code-diff proposal generator
  strategies/
    base.py              # Strategy ABC
    binary_forecast.py   # LLM-based up/down strategy
    mean_reversion.py    # Moving-average reversion strategy
    consensus.py         # Multi-strategy voting
  visual/
    sim.py               # Colony-style visualization state
  web/
    app.py               # FastAPI app and routes
    auth.py              # Password session middleware
    templates/
      dashboard.html     # Web UI
      login.html         # Auth page
config/
  borg.yaml              # Mission, loop timing, LLM, safety settings
  strategies.yaml        # Loaded trading strategies
monitoring/
  prometheus.yml         # Prometheus scrape config
  grafana/               # Dashboard + datasource provisioning
db/
  schema.sql             # Full PostgreSQL + pgvector schema
scripts/
  borg.service           # systemd unit
  profile_brain.py       # One-cycle latency profiler
tests/                   # pytest suite
setup.sh                 # One-shot Debian installer
docker-compose.yml       # Dev/prod container stack
Dockerfile
```

---

## Configuration

Settings are loaded from `config/borg.yaml` and can be overridden by environment variables or a `.env` file.

Key settings in `config/borg.yaml`:

| Path | Default | Description |
|------|---------|-------------|
| `symbols` | `BTC, ETH, SOL, HYPE` | Symbols to watch |
| `trading.confidence_threshold` | `65` | Minimum confidence to act |
| `trading.forecast_horizon_seconds` | `300` | Time before a forecast is resolved |
| `loop.interval_seconds` | `120` | Seconds between brain cycles |
| `loop.brain_concurrent_symbols` | `5` | Symbols forecast in parallel |
| `loop.self_improve_interval_seconds` | `1800` | Seconds between auto self-improvement attempts |
| `llm.model` | `qwen2:0.5b` | Generation model |
| `llm.force_fallback` | `true` | Skip Ollama; use fast deterministic rule engine |
| `llm.embed_model` | `nomic-embed-text:latest` | Embedding model |
| `llm.base_url` | `http://localhost:11434` | Ollama API base URL |
| `hyperlong.base_url` | `http://localhost:8080` | HyperLong dashboard API base URL |
| `web.port` | `8000` | Dashboard port |
| `safety.require_confirmation_for` | `self_modify, resource_heavy, clone, delete, assimilate, image_generation` | Actions needing approval |
| `smb.skip_patterns` | `node_modules, .git, ...` | Directories excluded from SMB inventory |

Example `.env`:

```bash
DATABASE_URL=postgresql+psycopg://borg:borg@localhost:5432/borg
BORG_PASSWORD=change_me
# LLM_BASE_URL=http://localhost:11434
# OPENAI_API_KEY=sk-...
```

---

## Using Ollama

1. Start Ollama:
   ```bash
   ollama serve
   ```
2. Pull models:
   ```bash
   ollama pull tinyllama:latest
   ollama pull nomic-embed-text:latest
   ```
3. Restart Borg. The dashboard will show **Ollama: reachable**.

If Ollama is not available, Borg transparently falls back to a deterministic rule engine for forecasts, embeddings, and chat answers.

---

## Hybrid LLM (Optional)

Borg can escalate uncertain forecasts to a more capable remote model. To enable it:

1. Add an OpenAI-compatible API key to `.env`:
   ```bash
   OPENAI_API_KEY=sk-...
   ```
2. Set the fallback provider in `config/borg.yaml`:
   ```yaml
   llm:
     fallback:
       provider: openai_compat
       base_url: https://api.openai.com/v1
       api_key_env: OPENAI_API_KEY
       model: gpt-4o-mini
   ```
3. Restart Borg.

When the local model returns confidence below `trading.confidence_threshold`, Borg asks the remote model for a second opinion. Stats are available at `/api/llm/stats`.

---

## Monitoring with Prometheus & Grafana

A Prometheus `/metrics` endpoint and a pre-built Grafana dashboard are included.

### During setup

```bash
./setup.sh --with-monitoring
```

This installs Prometheus and Grafana, configures them, binds them to all interfaces so they are reachable across the LAN, and starts the services.

- Prometheus UI: `http://<host-ip>:9090`
- Grafana UI: `http://<host-ip>:3000` (login: `admin` / `borg`)
- Borg dashboard: **Borg System Overview**

### Manual install/start (bare metal)

If you already ran setup without `--with-monitoring`:

```bash
./scripts/install_monitoring.sh   # install + configure
./scripts/start_monitoring.sh     # start services
```

Stop with:

```bash
./scripts/stop_monitoring.sh
```

### Docker Compose

If you run Borg via Docker Compose, Prometheus and Grafana are included:

```bash
docker compose up -d prometheus grafana
```

- Prometheus UI: `http://<host-ip>:9090`
- Grafana UI: `http://<host-ip>:3000` (login: `admin` / `${GRAFANA_PASSWORD:-admin}`)

### Configuration

- Borg metrics: `http://<host-ip>:8000/metrics`
- Bare-metal Prometheus config: `monitoring/prometheus.yml` (scrapes `localhost:8000`)
- Docker Compose Prometheus config: `monitoring/prometheus-docker.yml` (scrapes `borg:8000`)
- Grafana provisioning: `monitoring/grafana/`

Tracked metrics include brain-cycle duration, LLM inference latency, memory/CPU usage, forecasts generated, errors by component, and pending tasks.

---

## Databricks Export (Optional)

Borg can push its data to Databricks SQL/Delta tables for external dashboards and downstream analysis. The export runs once per day (after 07:00 UTC) and publishes:

- `borg_reports` — daily intelligence and system reports
- `borg_forecasts` — all forecasts with outcomes
- `borg_hip4_predictions` — HIP-4 binary-option predictions
- `borg_paper_trades` — paper-trade ledger and settlements
- `borg_candles` — latest market candles

To enable it, set the following in `.env`:

```bash
DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
DATABRICKS_WAREHOUSE_ID=<sql-warehouse-id>
```

Then set `databricks.enabled: true` in `config/borg.yaml` (default is `true`, but exports are skipped unless host/token are configured).

The table names and catalog/schema are configurable under `databricks:` in `config/borg.yaml`.

---

## Strategy Framework

Strategies are loaded from `config/strategies.yaml`. The default configuration includes:

- `binary_forecast`: LLM-based up/down/flat forecasting.
- `mean_reversion`: trades when price deviates from a moving average.
- `momentum`: rides established price momentum.
- `pairs_trading`: trades the spread between two correlated assets.
- `trend_following`: follows trends using moving averages and breakouts.
- `stat_arb`: statistical arbitrage using statistical models.
- `vwap`: volume-weighted average price execution signals.
- `twap`: time-weighted average price execution signals.
- `grid_trading`: signals at fixed price intervals around a central price.
- `market_making`: bid-ask spread signals.
- `dca`: dollar-cost averaging signals.
- `consensus_2vote`: only acts when 2+ sub-strategies agree.

To add a custom strategy, subclass `borg.strategies.base.Strategy`, implement `async def analyze(...)`, and add it to `config/strategies.yaml`.

---

## Profiling

Measure the latency of each brain-cycle phase:

```bash
PYTHONPATH=/home/theone/BorgDeploy .venv/bin/python scripts/profile_brain.py --symbol EURUSD
```

The script writes `profile_latest.json` with per-phase timings.

---

## Multi-Instance Coordination

Borg uses database-backed distributed locks (`borg/distributed.py`) so multiple instances can share work without duplicating market-data ingestion. Locks are stored in the `distributed_locks` table and expire automatically to prevent deadlocks.

---

## Report Marketplace

Borg replicates the saskpoly.xyz marketplace structure:

- **Daily Intelligence Brief** — overnight market deltas, news, MLB forecasts, prediction markets.
- **Home Run Tracker** — ranked candidates, weather scores, tiers, red flags.
- **Morning Brent Crude Brief** — Brent price, inventory signal, rig count, headlines.
- **Coffee News Edition** — newspaper-style layout with market tick, today's lock, and overnight summary.
- **Schedule** — upcoming macro, crude oil, and sports events.

Open the marketplace at `/reports/marketplace`. Click **Generate Today's Reports** to create the daily suite, or use `POST /api/reports/generate`. Each report has an HTML view and a downloadable PDF.

---

## Dashboard Guide

Open the dashboard at `http://<host-ip>:8000`.

If `BORG_PASSWORD` is set, you will be redirected to `/login`. Enter the password to continue.

The left sidebar provides links to all report pages and raw JSON API endpoints.

### Dashboard panels

| Panel | What it shows |
|-------|---------------|
| **Colony View** | Colony-style map. The green dot is the Borg agent; it moves between rooms as the brain changes phase (observe → plan → act → reflect). |
| **System Status** | Live CPU, memory, Ollama reachability, watched symbols, and last brain-cycle time. |
| **Current Task** | Current phase, symbol, task description, and the room the agent is heading to. |
| **Live Event Log** | Stream of everything Borg is doing. Buttons filter by category: All, Brain, Chat, System, Ingest. |
| **Talk to Borg** | Chat interface. Ask about status, forecasts, learnings, events, or the day/time. |
| **Active Goals** | Goals from the `goals` table. |
| **Recent Forecasts** | Latest up/down/flat forecasts with confidence and outcome (win/loss/pending). |
| **Self-Improvement** | Buttons to propose a code change or scan the input directory now. |

### Dashboard actions

| Button | Effect |
|--------|--------|
| **Seed Data** | Creates 30 minutes of synthetic candle history for each symbol. |
| **Run Cycle** | Runs one full OODA brain cycle. |
| **Propose Change** | Runs self-improvement analysis and creates a proposed code version. |
| **Scan Input** | Immediately scans `./input` and `/borg/input` for new files. |

---

## Reports & Data Views

All data is available both in the dashboard and via the API.

| Report | Dashboard location | API endpoint |
|--------|-------------------|--------------|
| System health | System Status panel | `/api/status` |
| Event log | Live Event Log panel | `/api/events` |
| Forecasts | Recent Forecasts panel | `/api/forecasts` |
| Learnings | (via chat) | `/api/learnings` |
| Goals | Active Goals panel | `/api/goals` |
| Tasks | (via API) | `/api/tasks` |
| Audit trail | (via API) | `/api/audit` |
| Code versions | Self-Improvement panel | `/api/versions` |
| Chat history | Talk to Borg panel | `/api/chat/history` |

---

## API Endpoints

### Read / status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard (auth-protected if `BORG_PASSWORD` set) |
| GET | `/healthz` | Health check |
| GET | `/api/status` | System status |
| GET | `/api/visual/state` | Colony map / agent state |
| GET | `/api/events` | Event log (`?limit=50&after_id=0&category=brain`) |
| GET | `/api/forecasts` | List forecasts (`?symbol=EURUSD&limit=50`) |
| GET | `/api/learnings` | Recent / search learnings (`?q=volatility&limit=10`) |
| GET | `/api/goals` | List goals (`?status=active&limit=50`) |
| GET | `/api/tasks` | List tasks (`?status=running&kind=forecast`) |
| GET | `/api/audit` | Audit log |
| GET | `/api/versions` | Proposed code versions |
| GET | `/api/chat/history` | Chat history (`?limit=50`) |
| GET | `/api/candles/{symbol}` | Latest candles (`?limit=50`) |
| GET | `/api/strategies` | Loaded strategies and risk metrics |
| GET | `/api/strategies/report` | Daily aggregated strategy signal counts |
| GET | `/api/hyperlong` | HyperLong data for all watched symbols |
| GET | `/api/hyperlong/{symbol}` | HyperLong chart/indicator data for one symbol |
| GET | `/api/inventory` | Inventory entries (`?status=scored&limit=100`) |
| GET | `/api/llm/stats` | Hybrid LLM usage statistics |
| GET | `/api/diary` | List generated daily diary files |
| GET | `/api/diary/{date}` | Markdown content of a diary entry |
| GET | `/api/diary/{date}/download` | Download diary Markdown file |
| GET | `/metrics` | Prometheus metrics |

### Write / control

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/login` | Submit `password` form to authenticate |
| GET | `/logout` | Clear session |
| POST | `/api/brain/seed` | Seed synthetic candle history |
| POST | `/api/brain/step` | Run one brain cycle manually |
| POST | `/api/candles` | Insert a validated candle (JSON body) |
| POST | `/api/candles/upload` | Upload CSV of candles (multipart `file`) |
| POST | `/api/forecasts` | Submit a manual forecast (JSON body) |
| POST | `/api/learnings` | Store a learning (`summary`, `detail`, `tags`) |
| POST | `/api/goals` | Create a goal (`title`, `description`, `priority`) |
| POST | `/api/goals/{id}/status` | Update goal status (`status=done`) |
| POST | `/api/self-improve` | Trigger self-improvement analysis |
| POST | `/api/ingest` | Scan input directories now |
| POST | `/api/versions/{id}/apply` | Apply a proposed version |
| POST | `/api/versions/{id}/reject` | Reject a proposed version (`reason=...`) |
| POST | `/api/safety/{action}/approve` | Approve a safety-gated action (e.g. `assimilate`) |
| POST | `/api/inventory/scan` | Scan SMB share (`root_path=hypelong`) |
| POST | `/api/inventory/score` | Score pending inventory entries |
| POST | `/api/inventory/{id}/stage` | Stage an entry for assimilation |
| POST | `/api/inventory/{id}/apply` | Apply a staged assimilation |
| POST | `/api/inventory/{id}/reject` | Reject a staged assimilation |
| POST | `/api/chat` | Chat with Borg (`message=...`) |
| POST | `/api/analyze` | Analyze a market summary (`symbol`, `summary`) |
| POST | `/api/events` | Emit a custom event (`message`, `category`, `level`) |
| POST | `/api/diary/generate` | Generate today's diary entry |

### API examples

```bash
BASE=http://localhost:8000

# Status
curl -s $BASE/api/status | python3 -m json.tool

# Seed data and run a cycle
curl -s -X POST $BASE/api/brain/seed
curl -s -X POST $BASE/api/brain/step

# Latest forecasts
curl -s "$BASE/api/forecasts?limit=5" | python3 -m json.tool

# Search memory
curl -s "$BASE/api/learnings?q=EURUSD&limit=3" | python3 -m json.tool

# Chat
curl -s -X POST $BASE/api/chat -d "message=what day is it" | python3 -m json.tool

# Propose a self-improvement change
curl -s -X POST $BASE/api/self-improve | python3 -m json.tool

# Apply version 1 (requires auth if BORG_PASSWORD is set)
curl -s -X POST $BASE/api/versions/1/apply

# Scan input directory
curl -s -X POST $BASE/api/ingest | python3 -m json.tool

# Upload candles CSV
curl -s -X POST $BASE/api/candles/upload -F "file=@candles.csv"
```

---

## Ingesting Data

Drop a CSV or JSON file into `./input` (or `/borg/input` if using Samba) with columns/keys:

```csv
symbol,ts,open,high,low,close,volume
EURUSD,2025-01-15T12:30:00Z,1.0840,1.0860,1.0820,1.0850,1500000
GBPUSD,2025-01-15T12:31:00Z,1.2700,1.2720,1.2680,1.2710,1200000
USDJPY,2025-01-15T12:32:00Z,151.50,151.80,151.20,151.60,1800000
```

The ingestion watcher scans every 5 seconds, validates rows with Pydantic, moves processed files to `./output/processed/`, and emits events.

### Samba shares

If you ran `./setup.sh` with Samba enabled, two guest shares are available:

- `\\<host-ip>\borg-input` — drop CSV/JSON files here.
- `\\<host-ip>\borg-output` — read exported reports here.

---

## Safety & Self-Improvement

Borg will **never** modify its own code without explicit approval. When `self_improve` proposes a change, it creates a version record with status `proposed`. Use the dashboard **Propose Change** / apply buttons or the `/api/versions/{id}/apply` endpoint to approve it. Every action is logged to `audit_log`.

The safety gate is controlled by `safety.require_confirmation_for` in `config/borg.yaml`:

```yaml
safety:
  require_confirmation_for:
    - self_modify
    - resource_heavy
    - clone
    - delete
```

---

## Running Tests

```bash
source .venv/bin/activate
pytest
```

Current status: **125 passed**.

---

## Operations

### Check if Borg is running

```bash
systemctl --user status borg
# or
lsof -i :8000
```

### View logs

```bash
journalctl --user -u borg -f
# or if running manually:
tail -f /home/theone/BorgDeploy/data/borg.log
```

### Restart

The service is managed by systemd as a user service:

```bash
systemctl --user restart borg
```

For a manual foreground session:

```bash
.venv/bin/python -m borg.main all --host 0.0.0.0 --port 8000
```

### Tune performance

If Ollama is slow or CPU is high:

- Increase `loop.interval_seconds` in `config/borg.yaml`.
- Stop Ollama to force the fast deterministic fallback.
- Reduce `symbols` to fewer markets.
- Set `resources.cpu_soft_limit_pct` / `resources.ram_soft_limit_pct` lower.

---

## Roadmap

This prototype implements the full Prompt-1 spec. See [`borg_improvements_essay.md`](./borg_improvements_essay.md) for a detailed roadmap covering:

1. Process pools to escape the GIL.
2. Distributed locks / event streaming for multi-host deployments.
3. Larger or hybrid LLMs for better forecast quality.
4. Modular strategy framework and data enrichment.
5. Fine-grained input validation and rate limiting.

For contributor conventions, see [`AGENTS.md`](./AGENTS.md).

---

## License

Prototype code — use at your own risk. Not financial advice.
