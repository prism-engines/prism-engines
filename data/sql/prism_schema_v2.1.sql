-- =============================================================================
-- PRISM Database Schema v2.1 (Consolidated Canonical)
-- =============================================================================
--
-- Changes from v2.0:
--   - Added: fred_code, category columns to indicators
--   - Added: system_name to systems table
--   - Merged: calibration_* and analysis_* tables (formerly in sql_schema_extension.py)
--   - Retained: timeseries as VIEW (not table) for backward compatibility
--
-- Architecture:
--   `indicator_values` is the single source of truth for all time series data.
--   All domains (market, economic, climate, etc.) use the same tables.
--   Legacy tables (market_prices, econ_values) retained for migration only.
--
-- =============================================================================

PRAGMA foreign_keys = ON;


-- =============================================================================
-- 1. SYSTEMS
-- =============================================================================
-- Registry of valid system/domain classifications.

CREATE TABLE IF NOT EXISTS systems (
    system      TEXT PRIMARY KEY,
    system_name TEXT DEFAULT ''
);

-- Preload default systems
INSERT OR IGNORE INTO systems (system, system_name) VALUES
    ('finance',     'Financial Markets'),
    ('market',      'Market Data'),
    ('economic',    'Economic Indicators'),
    ('climate',     'Climate/Weather'),
    ('biology',     'Biological Systems'),
    ('chemistry',   'Chemical Systems'),
    ('anthropology','Social/Demographic'),
    ('physics',     'Physical Systems'),
    ('benchmark',   'Benchmark Datasets');


-- =============================================================================
-- 2. INDICATORS
-- =============================================================================
-- Master registry of all indicators across all systems.
-- `name` is the universal key used throughout PRISM.

CREATE TABLE IF NOT EXISTS indicators (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,           -- Human-readable ID: 'SPY', 'DXY', 'M2SL'
    fred_code       TEXT,                           -- Optional: FRED API series ID
    system          TEXT NOT NULL DEFAULT 'market', -- Domain: 'market', 'economic', 'climate'
    category        TEXT,                           -- Grouping: 'Growth', 'Liquidity', 'Sentiment'
    frequency       TEXT NOT NULL DEFAULT 'daily',  -- 'daily', 'weekly', 'monthly'
    source          TEXT,                           -- 'yahoo', 'fred', 'stooq', 'tiingo'
    units           TEXT,
    description     TEXT,
    metadata        TEXT,                           -- JSON blob for extensibility
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),

    FOREIGN KEY (system) REFERENCES systems(system)
);

CREATE INDEX IF NOT EXISTS idx_indicators_system    ON indicators(system);
CREATE INDEX IF NOT EXISTS idx_indicators_name      ON indicators(name);
CREATE INDEX IF NOT EXISTS idx_indicators_category  ON indicators(category);
CREATE INDEX IF NOT EXISTS idx_indicators_source    ON indicators(source);
CREATE UNIQUE INDEX IF NOT EXISTS idx_indicators_fred_code ON indicators(fred_code);


-- =============================================================================
-- 3. INDICATOR_VALUES (THE UNIVERSAL DATA TABLE)
-- =============================================================================
-- Single source of truth for ALL time series across ALL domains.
--
-- Schema notes:
--   - quality_flag: 'verified', 'estimated', 'interpolated', 'missing'
--   - provenance: origin of data point ('fred', 'yahoo', 'calculated', 'migration')
--   - extra: JSON blob for ML outputs, plugin metadata

CREATE TABLE IF NOT EXISTS indicator_values (
    indicator_name  TEXT NOT NULL,              -- FK to indicators.name
    date            TEXT NOT NULL,              -- ISO format: YYYY-MM-DD
    value           REAL NOT NULL,

    quality_flag    TEXT DEFAULT NULL,
    provenance      TEXT DEFAULT NULL,
    extra           TEXT DEFAULT NULL,          -- JSON blob

    created_at      TEXT DEFAULT (datetime('now')),

    PRIMARY KEY (indicator_name, date),
    FOREIGN KEY (indicator_name) REFERENCES indicators(name)
);

CREATE INDEX IF NOT EXISTS idx_indicator_values_name       ON indicator_values(indicator_name);
CREATE INDEX IF NOT EXISTS idx_indicator_values_date       ON indicator_values(date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_name_date  ON indicator_values(indicator_name, date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_provenance ON indicator_values(provenance);


-- =============================================================================
-- 4. FETCH_LOG
-- =============================================================================
-- Unified logging for all fetch operations.

CREATE TABLE IF NOT EXISTS fetch_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_name  TEXT NOT NULL,
    source          TEXT NOT NULL,
    fetch_date      TEXT NOT NULL,              -- When fetch occurred (ISO datetime)
    rows_fetched    INTEGER,
    status          TEXT NOT NULL,              -- 'success', 'error', 'partial'
    error_message   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_indicator ON fetch_log(indicator_name);
CREATE INDEX IF NOT EXISTS idx_fetch_log_date      ON fetch_log(fetch_date);


-- =============================================================================
-- 5. METADATA
-- =============================================================================
-- Key-value store for system-wide settings and state.

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);


-- =============================================================================
-- 6. CALIBRATION TABLES
-- =============================================================================
-- Lens tuning, indicator curation, and configuration storage.

-- Calibration: Lens Weights & Performance
CREATE TABLE IF NOT EXISTS calibration_lenses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lens_name     TEXT UNIQUE,
    weight        REAL,
    hit_rate      REAL,
    avg_lead_time REAL,
    tier          INTEGER,
    use_lens      INTEGER DEFAULT 1,          -- 1 = active, 0 = inactive
    updated_at    TEXT
);

-- Calibration: Indicator Configuration
CREATE TABLE IF NOT EXISTS calibration_indicators (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator       TEXT UNIQUE,
    tier            INTEGER,
    score           REAL,
    optimal_window  INTEGER,
    indicator_type  TEXT,                     -- 'fast', 'medium', 'slow'
    use_indicator   INTEGER DEFAULT 1,        -- 1 = active, 0 = inactive
    redundant_to    TEXT,                     -- NULL or name of better indicator
    updated_at      TEXT
);

-- Calibration: Thresholds and Global Config
CREATE TABLE IF NOT EXISTS calibration_config (
    key        TEXT PRIMARY KEY,
    value      TEXT,                          -- JSON or scalar
    updated_at TEXT
);


-- =============================================================================
-- 7. ANALYSIS TABLES
-- =============================================================================
-- Results storage: rankings, signals, consensus events.

-- Analysis: Daily Rankings
CREATE TABLE IF NOT EXISTS analysis_rankings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT,
    analysis_type   TEXT,                     -- 'tuned', 'overnight', 'calibrated'
    indicator       TEXT,
    consensus_rank  REAL,
    tier            INTEGER,
    window_used     INTEGER,
    created_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_rankings_date      ON analysis_rankings(run_date);
CREATE INDEX IF NOT EXISTS idx_rankings_indicator ON analysis_rankings(indicator);

-- Analysis: Daily Signals
CREATE TABLE IF NOT EXISTS analysis_signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date       TEXT,
    indicator      TEXT,
    signal_name    TEXT,
    signal_meaning TEXT,
    rank           REAL,
    status         TEXT,                      -- 'DANGER', 'WARNING', 'NORMAL'
    created_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_date   ON analysis_signals(run_date);
CREATE INDEX IF NOT EXISTS idx_signals_status ON analysis_signals(status);

-- Analysis: Detected Consensus Events (Regime Breaks)
CREATE TABLE IF NOT EXISTS consensus_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date     TEXT,
    peak_consensus REAL,
    duration_days  INTEGER,
    detected_at    TEXT,
    notes          TEXT
);

-- Analysis: Historical Consensus Timeseries
CREATE TABLE IF NOT EXISTS consensus_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT,
    consensus_value REAL,
    window_size     INTEGER,
    created_at      TEXT,

    UNIQUE(date, window_size)
);

CREATE INDEX IF NOT EXISTS idx_consensus_date ON consensus_history(date);


-- =============================================================================
-- 8. DEPRECATED TABLES (Backward Compatibility Only)
-- =============================================================================
-- These tables are DEPRECATED. All new code must use indicator_values.
-- Retained temporarily for migration support.

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


-- =============================================================================
-- 9. BACKWARD COMPATIBILITY VIEWS
-- =============================================================================
-- Views for code expecting old table structures.
-- timeseries is a VIEW, not a table - single source of truth is indicator_values.

CREATE VIEW IF NOT EXISTS timeseries AS
SELECT
    indicator_name AS indicator,
    'market' AS system,
    date,
    value,
    created_at
FROM indicator_values;


-- =============================================================================
-- 10. SCHEMA VERSION
-- =============================================================================

INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', '2.1');


-- =============================================================================
-- END OF SCHEMA v2.1
-- =============================================================================
