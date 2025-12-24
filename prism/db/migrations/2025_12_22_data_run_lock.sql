-- PRISM v3: Data Run Lock Table
-- Prevents duplicate or overlapping data-phase runs

CREATE TABLE meta.data_run_lock (
    run_id      VARCHAR PRIMARY KEY,
    phase       VARCHAR NOT NULL DEFAULT 'data',  -- e.g. 'data', 'derived'
    domain      VARCHAR NOT NULL,
    mode        VARCHAR NOT NULL DEFAULT 'short', -- 'short' or 'full'
    locked_by   VARCHAR,                          -- e.g. 'data_phase_orchestrator'
    started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

