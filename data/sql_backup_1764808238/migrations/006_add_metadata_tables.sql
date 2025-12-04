-- Migration 006: Add Metadata Tables
-- ===================================
-- Creates additional metadata/tracking tables for database management.
--
-- Tables:
--   - schema_migrations: Tracks applied migrations
--   - data_quality_log: Logs data quality issues
--   - fetch_log: Tracks data fetch operations

-- -----------------------------------------------------------------------------
-- schema_migrations Table
-- -----------------------------------------------------------------------------
-- Tracks which migrations have been applied to the database.
-- This enables the migration runner to skip already-applied migrations.

CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,          -- Migration version (e.g., "002", "003")
    filename TEXT NOT NULL,                -- Migration filename
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT                          -- Optional SHA256 of migration file
);

-- Index for fast version lookups
CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);


-- -----------------------------------------------------------------------------
-- data_quality_log Table
-- -----------------------------------------------------------------------------
-- Logs data quality issues detected during import/validation.
-- Useful for debugging and tracking data integrity over time.

CREATE TABLE IF NOT EXISTS data_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,              -- Table with the issue
    record_id TEXT,                        -- ID of the affected record
    issue_type TEXT NOT NULL,              -- Type: missing_data, outlier, format_error, etc.
    description TEXT,                      -- Human-readable description
    severity TEXT DEFAULT 'warning',       -- info, warning, error, critical
    resolved INTEGER DEFAULT 0,            -- 0 = unresolved, 1 = resolved
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT
);

-- Indexes for filtering
CREATE INDEX IF NOT EXISTS idx_data_quality_log_table ON data_quality_log(table_name);
CREATE INDEX IF NOT EXISTS idx_data_quality_log_type ON data_quality_log(issue_type);
CREATE INDEX IF NOT EXISTS idx_data_quality_log_severity ON data_quality_log(severity);
CREATE INDEX IF NOT EXISTS idx_data_quality_log_resolved ON data_quality_log(resolved);


-- -----------------------------------------------------------------------------
-- fetch_log Table
-- -----------------------------------------------------------------------------
-- Tracks data fetch operations (from Yahoo, FRED, etc.)
-- Useful for debugging, audit trails, and rate limiting.

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,                  -- Data source: yahoo, fred, bls, etc.
    entity TEXT NOT NULL,                  -- Ticker or series code fetched
    operation TEXT NOT NULL,               -- fetch, update, backfill, etc.
    status TEXT NOT NULL,                  -- success, error, partial
    rows_fetched INTEGER DEFAULT 0,        -- Number of rows retrieved
    rows_inserted INTEGER DEFAULT 0,       -- Number of rows inserted
    rows_updated INTEGER DEFAULT 0,        -- Number of rows updated
    error_message TEXT,                    -- Error details if status = error
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER                    -- Duration in milliseconds
);

-- Indexes for filtering and analysis
CREATE INDEX IF NOT EXISTS idx_fetch_log_source ON fetch_log(source);
CREATE INDEX IF NOT EXISTS idx_fetch_log_entity ON fetch_log(entity);
CREATE INDEX IF NOT EXISTS idx_fetch_log_status ON fetch_log(status);
CREATE INDEX IF NOT EXISTS idx_fetch_log_started_at ON fetch_log(started_at);


-- -----------------------------------------------------------------------------
-- Useful views for monitoring
-- -----------------------------------------------------------------------------

-- Recent fetch operations
CREATE VIEW IF NOT EXISTS v_recent_fetches AS
SELECT
    id,
    source,
    entity,
    operation,
    status,
    rows_fetched,
    rows_inserted,
    rows_updated,
    started_at,
    duration_ms
FROM fetch_log
ORDER BY started_at DESC
LIMIT 100;

-- Fetch statistics by source
CREATE VIEW IF NOT EXISTS v_fetch_stats_by_source AS
SELECT
    source,
    COUNT(*) AS total_fetches,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS failed,
    SUM(rows_fetched) AS total_rows_fetched,
    AVG(duration_ms) AS avg_duration_ms
FROM fetch_log
GROUP BY source;

-- Unresolved data quality issues
CREATE VIEW IF NOT EXISTS v_unresolved_quality_issues AS
SELECT
    table_name,
    issue_type,
    severity,
    COUNT(*) AS count
FROM data_quality_log
WHERE resolved = 0
GROUP BY table_name, issue_type, severity
ORDER BY
    CASE severity
        WHEN 'critical' THEN 1
        WHEN 'error' THEN 2
        WHEN 'warning' THEN 3
        ELSE 4
    END,
    count DESC;


SELECT 'Migration 006 complete - metadata tables created' AS status;
