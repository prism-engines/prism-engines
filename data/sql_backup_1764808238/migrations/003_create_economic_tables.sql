-- Migration 003: Create Economic Tables
-- ======================================
-- Creates tables for economic data: series definitions, values, and metadata.
--
-- Tables:
--   - econ_series: Economic indicator definitions (code, name, frequency)
--   - econ_values: Time series values with revision tracking
--   - econ_meta: Fetch/revision metadata

-- -----------------------------------------------------------------------------
-- econ_series Table
-- -----------------------------------------------------------------------------
-- Reference table for economic series (FRED codes, etc.)

CREATE TABLE IF NOT EXISTS econ_series (
    code TEXT PRIMARY KEY,                 -- Unique series code (e.g., "GDP", "UNRATE")
    human_name TEXT,                       -- Human-readable name
    frequency TEXT,                        -- daily, monthly, quarterly, yearly
    source TEXT,                           -- Data source (FRED, BLS, etc.)
    notes TEXT,                            -- Additional notes
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);


-- -----------------------------------------------------------------------------
-- econ_values Table
-- -----------------------------------------------------------------------------
-- Stores economic time series data with revision tracking.
-- Economic data is often revised, so we track `revision_asof` for point-in-time analysis.

CREATE TABLE IF NOT EXISTS econ_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,                    -- FK to econ_series.code
    date TEXT NOT NULL,                    -- ISO format: YYYY-MM-DD (observation date)
    revision_asof TEXT NOT NULL,           -- ISO format: YYYY-MM-DD (when this value was published)
    value_raw REAL NOT NULL,               -- Raw value as published
    value_yoy REAL,                        -- Year-over-year change (optional)
    value_mom REAL,                        -- Month-over-month change (optional)
    value_z REAL,                          -- Normalized value (optional)
    value_log REAL,                        -- Log value (optional)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (code) REFERENCES econ_series(code) ON DELETE CASCADE,
    UNIQUE(code, date, revision_asof)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_econ_values_code ON econ_values(code);
CREATE INDEX IF NOT EXISTS idx_econ_values_date ON econ_values(date);
CREATE INDEX IF NOT EXISTS idx_econ_values_revision_asof ON econ_values(revision_asof);
CREATE INDEX IF NOT EXISTS idx_econ_values_code_date ON econ_values(code, date);


-- -----------------------------------------------------------------------------
-- econ_meta Table
-- -----------------------------------------------------------------------------
-- Tracks when series were last fetched and the most recent revision.

CREATE TABLE IF NOT EXISTS econ_meta (
    code TEXT PRIMARY KEY,                 -- FK to econ_series.code
    last_fetched TEXT,                     -- ISO format: YYYY-MM-DD
    last_revision_asof TEXT,               -- Most recent revision_asof value
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (code) REFERENCES econ_series(code) ON DELETE CASCADE
);


-- -----------------------------------------------------------------------------
-- Triggers: Update timestamps
-- -----------------------------------------------------------------------------

CREATE TRIGGER IF NOT EXISTS update_econ_series_timestamp
    AFTER UPDATE ON econ_series
    FOR EACH ROW
BEGIN
    UPDATE econ_series SET updated_at = CURRENT_TIMESTAMP WHERE code = OLD.code;
END;

CREATE TRIGGER IF NOT EXISTS update_econ_meta_timestamp
    AFTER UPDATE ON econ_meta
    FOR EACH ROW
BEGIN
    UPDATE econ_meta SET updated_at = CURRENT_TIMESTAMP WHERE code = OLD.code;
END;
