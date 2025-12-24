-- PRISM v3: Data Run Tracking Tables
-- Tracks run metadata and step progress

-- Run header/metadata
CREATE TABLE meta.data_runs (
    run_id              VARCHAR PRIMARY KEY,
    domain              VARCHAR NOT NULL,
    mode                VARCHAR NOT NULL DEFAULT 'short',  -- 'short' or 'full'
    started_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at        TIMESTAMP,
    status              VARCHAR DEFAULT 'running',  -- 'running', 'completed', 'failed'
    indicators_requested INTEGER,
    indicators_scanned   INTEGER,
    indicators_failed    INTEGER,
    error_summary        VARCHAR,
    notes                VARCHAR
);

-- Step-by-step execution log
CREATE TABLE meta.data_steps (
    run_id          VARCHAR NOT NULL,
    step_name       VARCHAR NOT NULL,
    status          VARCHAR NOT NULL,  -- 'started', 'completed', 'failed'
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP,
    n_items         INTEGER DEFAULT 0,
    error_message   VARCHAR,
    PRIMARY KEY (run_id, step_name)
);
