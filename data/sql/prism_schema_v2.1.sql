-- ============================================================================
-- PRISM Database Schema v2.1 (Canonical)
-- ============================================================================
--
-- This is the official, consolidated schema for PRISM Engine.
--
-- Key design decisions:
--   - `indicators.name` is the human-readable identifier (NOT indicator_name)
--   - `indicator_values.indicator_name` references `indicators.name`
--   - `timeseries` remains a VIEW for backward compatibility
--   - Calibration and analysis tables merged into main schema
--   - `fred_code` and `category` added to indicators
--   - `system_name` added to systems
--
-- Usage:
--   For new databases: sqlite3 prism.db < prism_schema_v2.1.sql
--   For existing databases: Use 003_v2.1_migration.sql
--
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ============================================================================
-- SYSTEMS TABLE
-- ============================================================================
-- Registry of valid system/domain classifications.

CREATE TABLE IF NOT EXISTS systems (
    system      TEXT PRIMARY KEY,
    system_name TEXT DEFAULT ''           -- Human-readable name
);

-- Preload default systems
INSERT OR IGNORE INTO systems(system, system_name) VALUES
    ('finance', 'Financial Markets'),
    ('market', 'Market Data'),
    ('economic', 'Economic Indicators'),
    ('climate', 'Climate Data'),
    ('biology', 'Biological Systems'),
    ('chemistry', 'Chemical Data'),
    ('anthropology', 'Anthropological Data'),
    ('physics', 'Physics Data'),
    ('benchmark', 'Benchmark Datasets');


-- ============================================================================
-- INDICATORS TABLE
-- ============================================================================
-- Master registry of all indicators across all systems.

CREATE TABLE IF NOT EXISTS indicators (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,         -- Human-readable ID (e.g., 'SPY', 'M2SL')
    fred_code   TEXT UNIQUE,                  -- FRED API series ID (e.g., 'FEDFUNDS')
    system      TEXT NOT NULL DEFAULT 'market',
    category    TEXT,                         -- Grouping: Growth, Liquidity, Sentiment, etc.
    frequency   TEXT NOT NULL DEFAULT 'daily',
    source      TEXT,                         -- e.g., 'yahoo', 'fred', 'tiingo'
    units       TEXT,
    description TEXT,
    metadata    TEXT,                         -- JSON blob for extensibility
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (system) REFERENCES systems(system)
);

CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);
CREATE INDEX IF NOT EXISTS idx_indicators_category ON indicators(category);
CREATE INDEX IF NOT EXISTS idx_indicators_fred_code ON indicators(fred_code);


-- ============================================================================
-- INDICATOR VALUES TABLE (Universal Time Series)
-- ============================================================================
-- Single source of truth for ALL time series data.

CREATE TABLE IF NOT EXISTS indicator_values (
    indicator_name TEXT NOT NULL,             -- References indicators.name
    date           TEXT NOT NULL,             -- ISO format: YYYY-MM-DD
    value          REAL NOT NULL,

    quality_flag   TEXT DEFAULT NULL,         -- verified/estimated/interpolated/missing
    provenance     TEXT DEFAULT NULL,         -- fred/yahoo/migration/calculated/benchmark
    extra          TEXT DEFAULT NULL,         -- JSON blob for ML outputs, plugin data

    created_at     TEXT DEFAULT (datetime('now')),

    PRIMARY KEY (indicator_name, date),
    FOREIGN KEY (indicator_name) REFERENCES indicators(name)
);

CREATE INDEX IF NOT EXISTS idx_indicator_values_name ON indicator_values(indicator_name);
CREATE INDEX IF NOT EXISTS idx_indicator_values_date ON indicator_values(date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_name_date ON indicator_values(indicator_name, date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_provenance ON indicator_values(provenance);


-- ============================================================================
-- TIMESERIES VIEW (Backward Compatibility)
-- ============================================================================
-- Maps old code paths to indicator_values. NOT a table.

CREATE VIEW IF NOT EXISTS timeseries AS
SELECT
    indicator_name AS indicator,
    'market' AS system,
    date,
    value,
    created_at
FROM indicator_values;


-- ============================================================================
-- FETCH LOG TABLE
-- ============================================================================
-- Tracks fetch operations for debugging and auditing.

CREATE TABLE IF NOT EXISTS fetch_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_name TEXT NOT NULL,
    source         TEXT NOT NULL,
    fetch_date     TEXT NOT NULL,
    rows_fetched   INTEGER,
    status         TEXT NOT NULL,             -- success/error/partial
    error_message  TEXT,
    created_at     TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_indicator ON fetch_log(indicator_name);
CREATE INDEX IF NOT EXISTS idx_fetch_log_date ON fetch_log(fetch_date);


-- ============================================================================
-- METADATA TABLE
-- ============================================================================
-- Key-value store for system-wide settings and state.

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Set schema version
INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', '2.1');


-- ============================================================================
-- LEGACY TABLES (Backward Compatibility)
-- ============================================================================
-- DEPRECATED: Use indicator_values instead. Kept for migration purposes.

CREATE TABLE IF NOT EXISTS market_prices (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    value  REAL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS econ_values (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    date      TEXT NOT NULL,
    value     REAL,
    UNIQUE(series_id, date)
);


-- ============================================================================
-- CALIBRATION TABLES (Lens Tuning System)
-- ============================================================================

CREATE TABLE IF NOT EXISTS calibration_lenses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lens_name     TEXT UNIQUE,
    weight        REAL,
    hit_rate      REAL,
    avg_lead_time REAL,
    tier          INTEGER,
    use_lens      INTEGER,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS calibration_indicators (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator      TEXT UNIQUE,
    tier           INTEGER,
    score          REAL,
    optimal_window INTEGER,
    indicator_type TEXT,
    use_indicator  INTEGER,
    redundant_to   TEXT,
    updated_at     TEXT
);

CREATE TABLE IF NOT EXISTS calibration_config (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);


-- ============================================================================
-- ANALYSIS TABLES (MRF, Rankings, Signals)
-- ============================================================================

CREATE TABLE IF NOT EXISTS analysis_rankings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date       TEXT,
    analysis_type  TEXT,
    indicator      TEXT,
    consensus_rank REAL,
    tier           INTEGER,
    window_used    INTEGER,
    created_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_rankings_date ON analysis_rankings(run_date);
CREATE INDEX IF NOT EXISTS idx_rankings_indicator ON analysis_rankings(indicator);

CREATE TABLE IF NOT EXISTS analysis_signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date       TEXT,
    indicator      TEXT,
    signal_name    TEXT,
    signal_meaning TEXT,
    rank           REAL,
    status         TEXT,
    created_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_date ON analysis_signals(run_date);
CREATE INDEX IF NOT EXISTS idx_signals_status ON analysis_signals(status);

CREATE TABLE IF NOT EXISTS consensus_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date     TEXT,
    peak_consensus REAL,
    duration_days  INTEGER,
    detected_at    TEXT,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS consensus_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT,
    consensus_value REAL,
    window_size     INTEGER,
    created_at      TEXT,
    UNIQUE(date, window_size)
);

CREATE INDEX IF NOT EXISTS idx_consensus_date ON consensus_history(date);


-- ============================================================================
-- END OF SCHEMA v2.1
-- ============================================================================
