-- PRISM Database Schema
-- ======================
-- SQLite schema for storing indicators and time series data.
--
-- Key concept:
--   `system` = top-level domain for an indicator (e.g., finance, climate, chemistry).
--   A single database can store multiple systems side by side.

-- -----------------------------------------------------------------------------
-- Indicators Table
-- -----------------------------------------------------------------------------
-- Stores metadata about each indicator (time series).

CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    system TEXT NOT NULL,              -- Domain: finance, climate, chemistry, etc.
    frequency TEXT NOT NULL,           -- daily, weekly, monthly, quarterly, yearly
    source TEXT,                       -- Data source (FRED, Yahoo, etc.)
    units TEXT,                        -- Unit of measurement
    description TEXT,                  -- Human-readable description
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookups by system (domain)
CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);

-- Index for fast lookups by name
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);


-- -----------------------------------------------------------------------------
-- Indicator Values Table
-- -----------------------------------------------------------------------------
-- Stores the actual time series data points.

CREATE TABLE IF NOT EXISTS indicator_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_id INTEGER NOT NULL,
    date TEXT NOT NULL,                -- ISO format: YYYY-MM-DD
    value REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (indicator_id) REFERENCES indicators(id) ON DELETE CASCADE,
    UNIQUE(indicator_id, date)         -- One value per indicator per date
);

-- Index for fast date range queries
CREATE INDEX IF NOT EXISTS idx_indicator_values_date ON indicator_values(date);

-- Index for fast lookups by indicator
CREATE INDEX IF NOT EXISTS idx_indicator_values_indicator_id ON indicator_values(indicator_id);


-- -----------------------------------------------------------------------------
-- Trigger: Update timestamps
-- -----------------------------------------------------------------------------
-- Automatically update `updated_at` when an indicator is modified.

CREATE TRIGGER IF NOT EXISTS update_indicator_timestamp
    AFTER UPDATE ON indicators
    FOR EACH ROW
BEGIN
    UPDATE indicators SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;


-- -----------------------------------------------------------------------------
-- Valid System Types (for reference)
-- -----------------------------------------------------------------------------
-- These are the allowed values for the `system` column:
--   - finance
--   - climate
--   - chemistry
--   - anthropology
--   - biology
--   - physics
--
-- Note: Validation is enforced at the application layer (prism_db.py).
