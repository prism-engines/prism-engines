-- PRISM Database Schema (Unified)
-- ================================
-- SQLite schema for storing indicators and time series data.
--
-- This is the SINGLE schema for all data types.
-- All market, economic, climate, etc. data uses the same tables.
--
-- Key concept:
--   `system` = top-level domain for an indicator (e.g., finance, climate).
--   A single database can store multiple systems side by side.

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

CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,             -- e.g., 'SPY', 'DXY', 'M2SL'
    system TEXT NOT NULL,                  -- e.g., 'finance', 'market', 'economic'
    frequency TEXT NOT NULL DEFAULT 'daily',
    source TEXT,                           -- e.g., 'yahoo', 'fred', 'stooq'
    units TEXT,
    description TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (system) REFERENCES systems(system)
);

CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);


-- -----------------------------------------------------------------------------
-- Indicator Values Table (THE primary data table)
-- -----------------------------------------------------------------------------
-- Stores all time series observations for all indicators.
-- Uses indicator name + system as composite key for simpler queries.

CREATE TABLE IF NOT EXISTS indicator_values (
    indicator TEXT NOT NULL,
    system TEXT NOT NULL,
    date TEXT NOT NULL,                    -- ISO format: YYYY-MM-DD
    value REAL NOT NULL,
    value_2 REAL,                          -- Optional secondary value (e.g., volume)
    adjusted_value REAL,                   -- Optional adjusted value
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (indicator, system, date)
);

CREATE INDEX IF NOT EXISTS idx_indicator_values_indicator
    ON indicator_values(indicator);

CREATE INDEX IF NOT EXISTS idx_indicator_values_system
    ON indicator_values(system);

CREATE INDEX IF NOT EXISTS idx_indicator_values_date
    ON indicator_values(date);

CREATE INDEX IF NOT EXISTS idx_indicator_values_indicator_date
    ON indicator_values(indicator, date);


-- -----------------------------------------------------------------------------
-- Fetch Log Table
-- -----------------------------------------------------------------------------
-- Tracks fetch operations for debugging and auditing.

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator TEXT NOT NULL,
    system TEXT NOT NULL,
    source TEXT NOT NULL,
    fetch_date TEXT NOT NULL,              -- When the fetch occurred
    rows_fetched INTEGER,
    status TEXT NOT NULL,                  -- 'success', 'error', 'partial'
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_indicator ON fetch_log(indicator);
CREATE INDEX IF NOT EXISTS idx_fetch_log_date ON fetch_log(fetch_date);


-- -----------------------------------------------------------------------------
-- Backward Compatibility View
-- -----------------------------------------------------------------------------
-- Creates a 'timeseries' view for any code expecting the old table name.

CREATE VIEW IF NOT EXISTS timeseries AS
SELECT
    indicator,
    system,
    date,
    value,
    created_at
FROM indicator_values;
