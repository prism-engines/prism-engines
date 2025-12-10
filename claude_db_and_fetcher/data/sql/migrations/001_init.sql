-- PRISM Database Migration 001: Schema v2 Initialization
-- ========================================================
-- This migration ensures Schema v2 tables exist.
-- Safe to re-run (uses IF NOT EXISTS throughout).

PRAGMA foreign_keys = ON;

-- Systems Table
CREATE TABLE IF NOT EXISTS systems (
    system TEXT PRIMARY KEY
);

INSERT OR IGNORE INTO systems(system) VALUES
    ('finance'),
    ('market'),
    ('economic'),
    ('climate'),
    ('biology'),
    ('chemistry'),
    ('anthropology'),
    ('physics');

-- Indicators Table
CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    system TEXT NOT NULL DEFAULT 'market',
    frequency TEXT NOT NULL DEFAULT 'daily',
    source TEXT,
    units TEXT,
    description TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (system) REFERENCES systems(system)
);

CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);

-- Indicator Values Table (Universal Data Table)
CREATE TABLE IF NOT EXISTS indicator_values (
    indicator_name TEXT NOT NULL,
    date           TEXT NOT NULL,
    value          REAL NOT NULL,
    quality_flag   TEXT DEFAULT NULL,
    provenance     TEXT DEFAULT NULL,
    extra          TEXT DEFAULT NULL,
    created_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (indicator_name, date),
    FOREIGN KEY (indicator_name) REFERENCES indicators(name)
);

CREATE INDEX IF NOT EXISTS idx_indicator_values_name ON indicator_values(indicator_name);
CREATE INDEX IF NOT EXISTS idx_indicator_values_date ON indicator_values(date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_name_date ON indicator_values(indicator_name, date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_provenance ON indicator_values(provenance);

-- Fetch Log Table
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_name TEXT NOT NULL,
    source TEXT NOT NULL,
    fetch_date TEXT NOT NULL,
    rows_fetched INTEGER,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_indicator ON fetch_log(indicator_name);
CREATE INDEX IF NOT EXISTS idx_fetch_log_date ON fetch_log(fetch_date);

-- Metadata Table
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Deprecated Tables (Backward Compatibility)
CREATE TABLE IF NOT EXISTS market_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS econ_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL,
    UNIQUE(series_id, date)
);

-- Backward Compatibility View
CREATE VIEW IF NOT EXISTS timeseries AS
SELECT
    indicator_name AS indicator,
    'market' AS system,
    date,
    value,
    created_at
FROM indicator_values;
