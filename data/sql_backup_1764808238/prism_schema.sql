-- PRISM Database Schema
-- ======================
-- SQLite schema for storing indicators and time series data.
--
-- Key concept:
--   `system` = top-level domain for an indicator (e.g., finance, climate, chemistry).
--   A single database can store multiple systems side by side.

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Indicators Table
-- -----------------------------------------------------------------------------
-- Stores metadata about each indicator (time series).

CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,             -- e.g., 'SPY', 'DXY', 'M2SL'
    system TEXT NOT NULL,                  -- e.g., 'finance', 'climate'
    frequency TEXT NOT NULL DEFAULT 'daily',
    source TEXT,
    units TEXT,
    description TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------------
-- Indicator Values Table (simplified storage)
-- -----------------------------------------------------------------------------
-- Stores the actual observations for each indicator.
-- Uses indicator name + system as composite key for simpler queries.

CREATE TABLE IF NOT EXISTS indicator_values (
    indicator TEXT NOT NULL,
    system TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (indicator, system, date)
);

CREATE INDEX IF NOT EXISTS idx_indicator_values_indicator
    ON indicator_values(indicator);

CREATE INDEX IF NOT EXISTS idx_indicator_values_system
    ON indicator_values(system);

CREATE INDEX IF NOT EXISTS idx_indicator_values_date
    ON indicator_values(date);

-- -----------------------------------------------------------------------------
-- Time Series Table (legacy/normalized storage)
-- -----------------------------------------------------------------------------
-- Stores the actual observations for each indicator with foreign key.

CREATE TABLE IF NOT EXISTS timeseries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    value REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (indicator_id) REFERENCES indicators(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_timeseries_indicator_date
    ON timeseries(indicator_id, date);

-- -----------------------------------------------------------------------------
-- System Registry Table
-- -----------------------------------------------------------------------------
-- Tracks the valid systems (domains) available for PRISM.

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
