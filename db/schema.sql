-- Borg full schema for PostgreSQL + pgvector
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Mission & state (KV store)
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Goals & Tasks (Planning layer)
CREATE TABLE IF NOT EXISTS goals (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    priority INTEGER DEFAULT 50,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','paused','done','abandoned')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ('analyze','forecast','ingest','self_improve','housekeeping','reflect')),
    payload JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','needs_confirm','running','done','failed','rejected')),
    result JSONB,
    created_by TEXT NOT NULL DEFAULT 'system',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal_id);

-- Market Data & Forecasts
CREATE TABLE IF NOT EXISTS market_candles (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON market_candles(symbol, ts DESC);

CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    symbol TEXT NOT NULL,
    horizon_s INTEGER NOT NULL DEFAULT 300,
    direction TEXT NOT NULL CHECK (direction IN ('up','down','flat')),
    confidence NUMERIC NOT NULL,
    rationale TEXT,
    features JSONB,
    outcome TEXT CHECK (outcome IN ('win','loss','pending','expired')),
    correct BOOLEAN,
    reasoning_output JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_forecasts_symbol ON forecasts(symbol);
CREATE INDEX IF NOT EXISTS idx_forecasts_unresolved ON forecasts(outcome) WHERE outcome = 'pending';
CREATE INDEX IF NOT EXISTS idx_forecasts_created ON forecasts(created_at DESC);

-- HIP-4 binary option predictions
CREATE TABLE IF NOT EXISTS hip4_predictions (
    id SERIAL PRIMARY KEY,
    underlying TEXT NOT NULL,
    outcome_id INTEGER NOT NULL,
    expiry TIMESTAMPTZ NOT NULL,
    target_price NUMERIC NOT NULL,
    yes_price NUMERIC NOT NULL,
    no_price NUMERIC NOT NULL,
    implied_probability NUMERIC NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('up','down','flat')),
    confidence NUMERIC NOT NULL,
    rationale TEXT,
    UNIQUE(underlying, expiry)
);

-- Daily HIP-4 paper trades (one per expiry date)
CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    underlying TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('up','down')),
    side TEXT NOT NULL CHECK (side IN ('YES','NO')),
    target_price NUMERIC NOT NULL,
    entry_price NUMERIC NOT NULL,
    token_price NUMERIC NOT NULL,
    quantity NUMERIC NOT NULL,
    stake NUMERIC NOT NULL,
    potential_payout NUMERIC NOT NULL,
    expiry TIMESTAMPTZ NOT NULL,
    outcome TEXT CHECK (outcome IN ('win','loss','pending')) DEFAULT 'pending',
    settle_price NUMERIC,
    pnl NUMERIC,
    settled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trade_date)
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_date ON paper_trades(trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_paper_trades_open ON paper_trades(outcome) WHERE outcome = 'pending';

-- Reflection & Learning
CREATE TABLE IF NOT EXISTS reflections (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    critique TEXT,
    score NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS learnings (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    summary TEXT NOT NULL,
    detail TEXT,
    embedding VECTOR(768),
    tags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_learnings_embed ON learnings USING hnsw (embedding vector_cosine_ops);

-- Memory & Consciousness
CREATE TABLE IF NOT EXISTS memory (
    id SERIAL PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('observation','fact','episode','summary')),
    content TEXT NOT NULL,
    embedding VECTOR(768),
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_memory_embed ON memory USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    actor TEXT NOT NULL,
    trigger TEXT,
    market_state JSONB,
    regime TEXT NOT NULL,
    trade_signal JSONB,
    executed BOOLEAN DEFAULT TRUE,
    outcome JSONB,
    reasoning_output JSONB,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_episodes_actor_regime ON episodes(actor, regime, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_embed ON episodes USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS reasoning_audits (
    id SERIAL PRIMARY KEY,
    forecast_id INTEGER REFERENCES forecasts(id) ON DELETE SET NULL,
    reasoning_decision TEXT,
    reasoning_confidence NUMERIC,
    reasoning_why TEXT,
    outcome_win BOOLEAN,
    outcome_pnl NUMERIC,
    calibration_error NUMERIC,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_reasoning_audits_created ON reasoning_audits(created_at DESC);

CREATE TABLE IF NOT EXISTS conscious_summaries (
    id SERIAL PRIMARY KEY,
    summary TEXT NOT NULL,
    context JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS consciousness_reports (
    id SERIAL PRIMARY KEY,
    period TEXT NOT NULL,
    report_date DATE NOT NULL,
    report_text TEXT NOT NULL,
    score NUMERIC,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_consciousness_reports_period_date ON consciousness_reports(period, report_date DESC);

-- Events
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ DEFAULT NOW(),
    level TEXT NOT NULL CHECK (level IN ('DEBUG','INFO','WARN','ERROR')),
    category TEXT NOT NULL,
    phase TEXT,
    message TEXT NOT NULL,
    symbol TEXT,
    metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);

-- Safety & Audit
CREATE TABLE IF NOT EXISTS versions (
    id SERIAL PRIMARY KEY,
    module TEXT NOT NULL,
    version TEXT NOT NULL,
    path TEXT,
    diff TEXT,
    new_content TEXT,
    status TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed','applied','rejected')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    applied_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_versions_module ON versions(module, created_at DESC);

CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    detail JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
    message TEXT NOT NULL,
    ts TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_conversations_ts ON conversations(ts DESC);

-- Brain cycles
CREATE TABLE IF NOT EXISTS brain_cycles (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT DEFAULT 'running',
    summary TEXT
);

-- Distributed locks (multi-instance coordination)
CREATE TABLE IF NOT EXISTS distributed_locks (
    lock_name TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    acquired_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_distributed_locks_expires ON distributed_locks(expires_at);

-- Report marketplace
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    report_date DATE NOT NULL,
    description TEXT,
    content_json JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'published' CHECK (status IN ('published','draft','archived')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_reports_category_date ON reports(category, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_reports_slug ON reports(slug);

CREATE TABLE IF NOT EXISTS market_deltas (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    price NUMERIC NOT NULL,
    change_pct NUMERIC NOT NULL,
    category TEXT NOT NULL,
    report_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_market_deltas_date ON market_deltas(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_market_deltas_symbol ON market_deltas(symbol);

CREATE TABLE IF NOT EXISTS scheduled_events (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    impact TEXT NOT NULL DEFAULT 'Medium',
    event_time TIMESTAMPTZ NOT NULL,
    description TEXT,
    tags TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scheduled_events_time ON scheduled_events(event_time);
CREATE INDEX IF NOT EXISTS idx_scheduled_events_category ON scheduled_events(category);

CREATE TABLE IF NOT EXISTS inventory_entries (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    name TEXT,
    entry_type TEXT NOT NULL CHECK (entry_type IN ('file','dir')),
    size_bytes BIGINT,
    content_hash TEXT,
    content TEXT,
    metadata JSONB,
    assimilation_score NUMERIC,
    assimilation_status TEXT NOT NULL DEFAULT 'pending' CHECK (assimilation_status IN ('pending','scored','staged','approved','rejected','applied')),
    assimilation_reason TEXT,
    version_id INTEGER REFERENCES versions(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_inventory_source ON inventory_entries(source);
CREATE INDEX IF NOT EXISTS idx_inventory_status ON inventory_entries(assimilation_status);
CREATE INDEX IF NOT EXISTS idx_inventory_score ON inventory_entries(assimilation_score DESC);

-- Seed mission
INSERT INTO kv (key, value) VALUES
('mission', '{"text": "Assist the user with autonomous binary-options market analysis, learning from outcomes, and improving forecasts while respecting safety guardrails."}'::jsonb)
ON CONFLICT (key) DO NOTHING;
