-- 2025_12_22_fetch_tracking.sql
-- Fetch tracking tables for FetchRunner
--
-- Required by: prism/fetch/runner.py
-- These tables track fetch run metadata and per-indicator results.


-- Fetch run tracking
CREATE TABLE meta.fetch_runs (
    run_id              VARCHAR PRIMARY KEY,
    started_at          TIMESTAMP NOT NULL,
    completed_at        TIMESTAMP,
    status              VARCHAR NOT NULL,   -- 'running', 'completed', 'failed'
    indicators_requested INTEGER,
    indicators_succeeded INTEGER,
    indicators_failed   INTEGER,
    error_message       VARCHAR
);

-- Per-indicator fetch results
CREATE TABLE meta.fetch_results (
    run_id          VARCHAR NOT NULL,
    indicator_id    VARCHAR NOT NULL,
    source          VARCHAR NOT NULL,
    status          VARCHAR NOT NULL,       -- 'success', 'failed', 'skipped'
    rows_fetched    INTEGER,
    first_date      DATE,
    last_date       DATE,
    error_message   VARCHAR,
    fetched_at      TIMESTAMP NOT NULL,

    PRIMARY KEY (run_id, indicator_id)
);

-- Index for querying by indicator
CREATE INDEX IF NOT EXISTS idx_fetch_results_indicator
    ON meta.fetch_results(indicator_id);
