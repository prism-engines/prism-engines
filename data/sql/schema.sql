-- PRISM Database Schema v2 (Panel-Agnostic)
-- ==========================================
-- Universal data model where all indicators share one time-series table.
--
-- Prepares PRISM for:
--   - Modular panels (market, economy, climate, etc.)
--   - Unified fetch system
--   - Plugin engines
--   - Multi-domain data
--   - ML/Meta feedback storage
--
-- Key concept:
--   `indicator_name` is the universal key that links indicators to indicator_values.
--   All domains (market, economic, climate, biology, etc.) use the same tables.

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Systems Table
-- -----------------------------------------------------------------------------
-- Tracks the valid systems (domains) available.

CREATE TABLE IF NOT EXISTS systems (
    system TEXT PRIMARY KEY
);

-- Preload default systems
INSERT OR IGNORE INTO systems(system) VALUES
    ('finance'),
    ('market'),
    ('economic'),
    ('climate'),
    ('biology'),
    ('chemistry'),
    ('anthropology'),
    ('physics');


-- -----------------------------------------------------------------------------
-- Indicators Table
-- -----------------------------------------------------------------------------
-- Stores metadata about each indicator (time series).
-- This is the registry of all known indicators.

CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,             -- e.g., 'SPY', 'DXY', 'M2SL', 'co2_level'
    system TEXT NOT NULL DEFAULT 'market', -- e.g., 'finance', 'market', 'economic', 'climate'
    frequency TEXT NOT NULL DEFAULT 'daily',
    source TEXT,                           -- e.g., 'yahoo', 'fred', 'stooq', 'noaa'
    units TEXT,
    description TEXT,
    metadata TEXT,                         -- JSON metadata for extensibility
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (system) REFERENCES systems(system)
);

CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);


-- -----------------------------------------------------------------------------
-- Indicator Values Table (THE UNIVERSAL DATA TABLE)
-- -----------------------------------------------------------------------------
-- Stores ALL time series observations for ALL indicators across ALL domains.
-- This is the single source of truth for all PRISM data.
--
-- Schema v2 additions:
--   - quality_flag: data quality indicator (e.g., 'verified', 'estimated', 'missing')
--   - provenance: origin/source of this specific data point
--   - extra: JSON blob for ML outputs, metadata, plugin-specific data

CREATE TABLE IF NOT EXISTS indicator_values (
    indicator_name TEXT NOT NULL,          -- References indicators.name
    date           TEXT NOT NULL,          -- ISO format: YYYY-MM-DD
    value          REAL NOT NULL,

    quality_flag   TEXT DEFAULT NULL,      -- 'verified', 'estimated', 'interpolated', 'missing'
    provenance     TEXT DEFAULT NULL,      -- Source/origin: 'fred', 'yahoo', 'migration', 'calculated'
    extra          TEXT DEFAULT NULL,      -- JSON blob for ML/Meta outputs, plugin data

    created_at     TEXT DEFAULT (datetime('now')),

    PRIMARY KEY (indicator_name, date),
    FOREIGN KEY (indicator_name) REFERENCES indicators(name)
);

CREATE INDEX IF NOT EXISTS idx_indicator_values_name
    ON indicator_values(indicator_name);

CREATE INDEX IF NOT EXISTS idx_indicator_values_date
    ON indicator_values(date);

CREATE INDEX IF NOT EXISTS idx_indicator_values_name_date
    ON indicator_values(indicator_name, date);

CREATE INDEX IF NOT EXISTS idx_indicator_values_provenance
    ON indicator_values(provenance);


-- -----------------------------------------------------------------------------
-- Fetch Log Table
-- -----------------------------------------------------------------------------
-- Tracks fetch operations for debugging and auditing.

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_name TEXT NOT NULL,
    source TEXT NOT NULL,
    fetch_date TEXT NOT NULL,              -- When the fetch occurred
    rows_fetched INTEGER,
    status TEXT NOT NULL,                  -- 'success', 'error', 'partial'
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_indicator ON fetch_log(indicator_name);
CREATE INDEX IF NOT EXISTS idx_fetch_log_date ON fetch_log(fetch_date);


-- -----------------------------------------------------------------------------
-- Metadata Table
-- -----------------------------------------------------------------------------
-- Key-value store for system-wide settings and state.

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);


-- -----------------------------------------------------------------------------
-- DEPRECATED TABLES (Backward Compatibility)
-- -----------------------------------------------------------------------------
-- These tables are DEPRECATED and will be removed in a future version.
-- All new code should use indicator_values instead.
-- Kept temporarily for migration and backward compatibility.

-- DEPRECATED: Use indicator_values instead
CREATE TABLE IF NOT EXISTS market_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL,
    UNIQUE(ticker, date)
);

-- DEPRECATED: Use indicator_values instead
CREATE TABLE IF NOT EXISTS econ_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL,
    UNIQUE(series_id, date)
);


-- -----------------------------------------------------------------------------
-- Backward Compatibility Views
-- -----------------------------------------------------------------------------
-- Creates views for code expecting old table structures.

-- View that maps old indicator_values schema to new
CREATE VIEW IF NOT EXISTS timeseries AS
SELECT
    indicator_name AS indicator,
    'market' AS system,  -- Default system for backward compat
    date,
    value,
    created_at
FROM indicator_values;
