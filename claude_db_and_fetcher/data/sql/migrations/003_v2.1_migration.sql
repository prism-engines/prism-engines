-- =============================================================================
-- PRISM Migration: v2.0 → v2.1
-- =============================================================================
--
-- Run this ONCE on existing databases to upgrade to schema v2.1.
-- Safe to run multiple times (uses IF NOT EXISTS guards).
--
-- IMPORTANT: SQLite ALTER TABLE limitations
--   - Cannot add UNIQUE constraint directly via ALTER TABLE
--   - Must use separate CREATE UNIQUE INDEX statement
--   - Cannot add NOT NULL without DEFAULT
--
-- Usage:
--   sqlite3 prism.db < 003_v2.1_migration.sql
--
-- =============================================================================


-- =============================================================================
-- 1. INDICATORS TABLE UPDATES
-- =============================================================================

-- Add fred_code column (no UNIQUE here - SQLite limitation)
-- We'll add uniqueness via index below
ALTER TABLE indicators ADD COLUMN fred_code TEXT;

-- Add category column
ALTER TABLE indicators ADD COLUMN category TEXT;

-- Add unique index on fred_code (this enforces uniqueness)
CREATE UNIQUE INDEX IF NOT EXISTS idx_indicators_fred_code ON indicators(fred_code);

-- Add index on category for fast grouping queries
CREATE INDEX IF NOT EXISTS idx_indicators_category ON indicators(category);

-- Add index on source if missing
CREATE INDEX IF NOT EXISTS idx_indicators_source ON indicators(source);


-- =============================================================================
-- 2. SYSTEMS TABLE UPDATE
-- =============================================================================

-- Add system_name column for human-readable names
ALTER TABLE systems ADD COLUMN system_name TEXT DEFAULT '';

-- Update existing systems with friendly names
UPDATE systems SET system_name = 'Financial Markets' WHERE system = 'finance' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Market Data' WHERE system = 'market' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Economic Indicators' WHERE system = 'economic' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Climate/Weather' WHERE system = 'climate' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Biological Systems' WHERE system = 'biology' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Chemical Systems' WHERE system = 'chemistry' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Social/Demographic' WHERE system = 'anthropology' AND (system_name IS NULL OR system_name = '');
UPDATE systems SET system_name = 'Physical Systems' WHERE system = 'physics' AND (system_name IS NULL OR system_name = '');


-- =============================================================================
-- 3. CALIBRATION TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS calibration_lenses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lens_name     TEXT UNIQUE,
    weight        REAL,
    hit_rate      REAL,
    avg_lead_time REAL,
    tier          INTEGER,
    use_lens      INTEGER DEFAULT 1,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS calibration_indicators (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator       TEXT UNIQUE,
    tier            INTEGER,
    score           REAL,
    optimal_window  INTEGER,
    indicator_type  TEXT,
    use_indicator   INTEGER DEFAULT 1,
    redundant_to    TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS calibration_config (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT
);


-- =============================================================================
-- 4. ANALYSIS TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS analysis_rankings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT,
    analysis_type   TEXT,
    indicator       TEXT,
    consensus_rank  REAL,
    tier            INTEGER,
    window_used     INTEGER,
    created_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_rankings_date      ON analysis_rankings(run_date);
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

CREATE INDEX IF NOT EXISTS idx_signals_date   ON analysis_signals(run_date);
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


-- =============================================================================
-- 5. METADATA TABLE (ensure exists)
-- =============================================================================

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);


-- =============================================================================
-- 6. RECORD MIGRATION
-- =============================================================================

INSERT OR REPLACE INTO metadata (key, value)
VALUES ('schema_version', '2.1');

INSERT OR REPLACE INTO metadata (key, value)
VALUES ('last_migration', datetime('now'));


-- =============================================================================
-- MIGRATION COMPLETE
-- =============================================================================
