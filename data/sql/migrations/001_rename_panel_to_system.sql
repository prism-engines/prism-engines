-- Migration 001: Rename `panel` column to `system` in indicators table
-- ======================================================================
-- This migration renames the `panel` column to `system` to better reflect
-- its purpose as the top-level domain for indicators (finance, climate, etc.).
--
-- SQLite Compatibility:
-- - SQLite 3.25.0+ supports ALTER TABLE ... RENAME COLUMN
-- - For older versions, we use the safe "create/copy/rename" pattern
--
-- This script uses the safe pattern for maximum compatibility.

-- Step 1: Disable foreign keys temporarily
PRAGMA foreign_keys=OFF;

-- Step 2: Begin transaction
BEGIN TRANSACTION;

-- Step 3: Create new indicators table with `system` column
CREATE TABLE IF NOT EXISTS indicators_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    system TEXT NOT NULL,              -- Renamed from `panel`
    frequency TEXT NOT NULL,
    source TEXT,
    units TEXT,
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Step 4: Copy data from old table (panel -> system)
INSERT INTO indicators_new (id, name, system, frequency, source, units, description, created_at, updated_at)
SELECT id, name, panel, frequency, source, units, description, created_at, updated_at
FROM indicators;

-- Step 5: Drop old table
DROP TABLE indicators;

-- Step 6: Rename new table to original name
ALTER TABLE indicators_new RENAME TO indicators;

-- Step 7: Recreate indexes
CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name);

-- Step 8: Recreate trigger for updating timestamps
CREATE TRIGGER IF NOT EXISTS update_indicator_timestamp
    AFTER UPDATE ON indicators
    FOR EACH ROW
BEGIN
    UPDATE indicators SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- Step 9: Commit transaction
COMMIT;

-- Step 10: Re-enable foreign keys
PRAGMA foreign_keys=ON;

-- Step 11: Verify migration
-- Run this query to confirm the column was renamed:
-- SELECT sql FROM sqlite_master WHERE type='table' AND name='indicators';
