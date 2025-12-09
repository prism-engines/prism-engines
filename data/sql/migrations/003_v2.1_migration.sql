-- ============================================================================
-- PRISM Schema v2.1 Migration
-- ============================================================================
--
-- Migrates existing databases to v2.1 schema.
--
-- Changes:
--   1. Add fred_code, category columns to indicators
--   2. Add system_name column to systems
--   3. Create calibration tables if missing
--   4. Create analysis tables if missing
--   5. Update schema version metadata
--
-- Usage:
--   sqlite3 prism.db < 003_v2.1_migration.sql
--   OR use migrate_to_v2.1.py for safer migration
--
-- ============================================================================

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- ============================================================================
-- 1. ADD COLUMNS TO INDICATORS TABLE
-- ============================================================================

-- Add fred_code column (ignore if exists)
ALTER TABLE indicators ADD COLUMN fred_code TEXT UNIQUE;

-- Add category column (ignore if exists)
ALTER TABLE indicators ADD COLUMN category TEXT;

-- Create index on category
CREATE INDEX IF NOT EXISTS idx_indicators_category ON indicators(category);

-- Create index on fred_code
CREATE INDEX IF NOT EXISTS idx_indicators_fred_code ON indicators(fred_code);


-- ============================================================================
-- 2. ADD system_name TO SYSTEMS TABLE
-- ============================================================================

ALTER TABLE systems ADD COLUMN system_name TEXT DEFAULT '';

-- Update existing systems with human-readable names
UPDATE systems SET system_name = 'Financial Markets' WHERE system = 'finance' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Market Data' WHERE system = 'market' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Economic Indicators' WHERE system = 'economic' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Climate Data' WHERE system = 'climate' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Benchmark Datasets' WHERE system = 'benchmark' AND (system_name IS NULL OR system_name = '');


-- ============================================================================
-- 3. CREATE CALIBRATION TABLES
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
-- 4. CREATE ANALYSIS TABLES
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
-- 5. UPDATE SCHEMA VERSION
-- ============================================================================

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);

INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', '2.1');
INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_migration', '003_v2.1_migration.sql');
INSERT OR REPLACE INTO metadata(key, value) VALUES ('migration_date', datetime('now'));


-- ============================================================================
-- END OF MIGRATION
-- ============================================================================
