-- PRISM Database Migration 001: Schema v2 Initialization
-- ========================================================
-- This migration ensures Schema v2 tables exist.
-- Safe to re-run (uses IF NOT EXISTS throughout).

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Systems Table
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS systems (
    system TEXT PRIMARY KEY
);

-- Preload default systems (no description column!)
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
CREATE INDEX IF NOT EXISTS idx_indicators_name   ON indicators(name);

-- -----------------------------------------------------------------------------
-- Indicator Values Table (Universal Time Series Table)
-- -----------------------------------------------------------------------------
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

CREATE INDEX IF NOT EXISTS idx_indicator_values_name       ON indicator_values(indicator_name);
CREATE INDEX IF NOT EXISTS idx_indicator_values_date       ON indicator_values(date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_name_date  ON indicator_values(indicator_name, date);
CREATE INDEX IF NOT EXISTS idx_indicator_values_provenance ON indicator_values(provenance);

-- -----------------------------------------------------------------------------
-- Fetch Log Table
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_name TEXT,
    source TEXT,
    timestamp TEXT,
    status TEXT,
    message TEXT
);
