-- =============================================================================
-- PRISM Database Schema
-- =============================================================================
--
-- DuckDB schema definitions for PRISM data storage.
-- All data lives in a single DuckDB file: data/prism.duckdb
--
-- CANONICAL PHASE MODEL:
--   data      - fetched + cleaned + admissible indicator data
--   derived   - first-order engine measurements (metrics of indicators)
--   structure - geometric organization (coherence, persistence, agreement)
--   binding   - final attachment / positioning within structure
--   meta      - orchestration, audits, windows
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- SCHEMAS (CANONICAL - NO OTHERS ALLOWED)
-- -----------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS data;       -- Fetched + cleaned + admissible
CREATE SCHEMA IF NOT EXISTS derived;    -- Engine math outputs (first-order measurements)
CREATE SCHEMA IF NOT EXISTS structure;  -- Geometric organization
CREATE SCHEMA IF NOT EXISTS binding;    -- Final attachment / positioning
CREATE SCHEMA IF NOT EXISTS meta;       -- Orchestration, audits, windows


-- -----------------------------------------------------------------------------
-- DATA SCHEMA: Fetched and cleaned indicator data
-- -----------------------------------------------------------------------------
-- Contains raw fetched data and cleaned/normalized versions
-- Engines read from data.indicators

-- Raw fetched data (archive layer)
CREATE TABLE IF NOT EXISTS data.raw_indicators (
    indicator_id    VARCHAR NOT NULL,
    date            DATE NOT NULL,
    value           DOUBLE,

    -- Metadata
    source          VARCHAR NOT NULL,       -- 'fred', 'tiingo', 'stooq'
    frequency       VARCHAR,                -- 'daily', 'weekly', 'monthly', 'quarterly'
    fetched_at      TIMESTAMP NOT NULL,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_id, date)
);

CREATE INDEX IF NOT EXISTS idx_data_raw_indicator
    ON data.raw_indicators (indicator_id);
CREATE INDEX IF NOT EXISTS idx_data_raw_date
    ON data.raw_indicators (date);

-- Cleaned indicator data (what engines read)
CREATE TABLE IF NOT EXISTS data.indicators (
    indicator_id    VARCHAR NOT NULL,
    date            DATE NOT NULL,
    value           DOUBLE,
    source          VARCHAR,
    frequency       VARCHAR,
    cleaning_method VARCHAR,                -- References cleaning_rules.md
    fetched_at      TIMESTAMP,
    cleaned_at      TIMESTAMP NOT NULL,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_id, date)
);

CREATE INDEX IF NOT EXISTS idx_data_indicator
    ON data.indicators (indicator_id);
CREATE INDEX IF NOT EXISTS idx_data_date
    ON data.indicators (date);


-- -----------------------------------------------------------------------------
-- META SCHEMA: Tracking and audit
-- -----------------------------------------------------------------------------

-- Fetch run tracking
CREATE TABLE IF NOT EXISTS meta.fetch_runs (
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
CREATE TABLE IF NOT EXISTS meta.fetch_results (
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

-- Indicator registry
CREATE TABLE IF NOT EXISTS meta.indicators (
    indicator_id    VARCHAR PRIMARY KEY,
    name            VARCHAR,
    source          VARCHAR NOT NULL,
    frequency       VARCHAR,
    category        VARCHAR,
    description     VARCHAR,
    units           VARCHAR,
    registered_at   TIMESTAMP NOT NULL
);

-- Engine run tracking
CREATE TABLE IF NOT EXISTS meta.engine_runs (
    run_id          VARCHAR PRIMARY KEY,
    engine_name     VARCHAR NOT NULL,       -- 'pca', 'correlation', 'granger', etc.
    phase           VARCHAR NOT NULL,       -- 'derived', 'structure', 'binding'
    started_at      TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    status          VARCHAR NOT NULL,       -- 'running', 'completed', 'failed'
    window_start    DATE,
    window_end      DATE,
    normalization   VARCHAR,                -- 'zscore', 'minmax', 'none', etc.
    parameters      VARCHAR,                -- JSON string of engine params
    error_message   VARCHAR
);

-- Data phase run lock (prevents duplicate runs, ensures immutability)
CREATE TABLE IF NOT EXISTS meta.data_run_lock (
    run_id      VARCHAR PRIMARY KEY,
    started_at  TIMESTAMP NOT NULL,
    domain      VARCHAR NOT NULL,
    mode        VARCHAR NOT NULL,
    locked_by   VARCHAR
);

-- Data phase run tracking
CREATE TABLE IF NOT EXISTS meta.data_runs (
    run_id              VARCHAR PRIMARY KEY,
    domain              VARCHAR NOT NULL,
    mode                VARCHAR NOT NULL,
    started_at          TIMESTAMP NOT NULL,
    completed_at        TIMESTAMP,
    status              VARCHAR,            -- 'running', 'completed', 'failed'
    indicators_requested INTEGER,
    indicators_scanned  INTEGER,
    notes               VARCHAR
);

-- Data phase step tracking
CREATE TABLE IF NOT EXISTS meta.data_steps (
    run_id          VARCHAR NOT NULL,
    step_name       VARCHAR NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    status          VARCHAR,                -- 'running', 'success', 'failed'
    n_items         INTEGER,
    error_message   VARCHAR,

    PRIMARY KEY (run_id, step_name)
);

-- Data phase reports
CREATE TABLE IF NOT EXISTS meta.data_reports (
    run_id          VARCHAR NOT NULL,
    report_path     VARCHAR NOT NULL,
    sha256          VARCHAR NOT NULL,
    report_markdown VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (run_id, report_path)
);


-- -----------------------------------------------------------------------------
-- DERIVED SCHEMA: First-order engine measurements
-- -----------------------------------------------------------------------------
-- Individual indicator analysis - metrics of indicators
-- No structure implied at this phase

CREATE TABLE IF NOT EXISTS derived.geometry_fingerprints (
    indicator_id    VARCHAR NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    dimension       VARCHAR NOT NULL,       -- 'curvature', 'velocity', 'volatility', etc.
    value           DOUBLE,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_id, window_start, window_end, dimension)
);

CREATE TABLE IF NOT EXISTS derived.pca_loadings (
    indicator_id    VARCHAR NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    component       INTEGER NOT NULL,       -- PC1, PC2, etc.
    loading         DOUBLE,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_id, window_start, window_end, component)
);

CREATE TABLE IF NOT EXISTS derived.pca_variance (
    window_start        DATE NOT NULL,
    window_end          DATE NOT NULL,
    component           INTEGER NOT NULL,
    variance_explained  DOUBLE,
    cumulative_variance DOUBLE,
    run_id              VARCHAR NOT NULL,

    PRIMARY KEY (window_start, window_end, component)
);

CREATE TABLE IF NOT EXISTS derived.correlation_matrix (
    indicator_id_1  VARCHAR NOT NULL,
    indicator_id_2  VARCHAR NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    correlation     DOUBLE,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_id_1, indicator_id_2, window_start, window_end)
);

CREATE TABLE IF NOT EXISTS derived.granger_causality (
    indicator_from  VARCHAR NOT NULL,
    indicator_to    VARCHAR NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    lag             INTEGER NOT NULL,
    f_statistic     DOUBLE,
    p_value         DOUBLE,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_from, indicator_to, window_start, window_end, lag)
);


-- -----------------------------------------------------------------------------
-- STRUCTURE SCHEMA: Geometric organization
-- -----------------------------------------------------------------------------
-- System-level geometry emerges from indicator relationships
-- Coherence, persistence, agreement, trajectories

CREATE TABLE IF NOT EXISTS structure.regimes (
    regime_id       VARCHAR NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    label           VARCHAR,                -- 'crisis', 'expansion', 'transition', etc.
    confidence      DOUBLE,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (regime_id, window_start, window_end)
);

CREATE TABLE IF NOT EXISTS structure.clusters (
    cluster_id      VARCHAR NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    indicator_id    VARCHAR NOT NULL,
    membership      DOUBLE,                 -- Soft clustering: 0-1
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (cluster_id, window_start, window_end, indicator_id)
);

CREATE TABLE IF NOT EXISTS structure.centroids (
    cluster_id      VARCHAR NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    dimension       VARCHAR NOT NULL,
    value           DOUBLE,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (cluster_id, window_start, window_end, dimension)
);

CREATE TABLE IF NOT EXISTS structure.similarity_matrix (
    window_start_1  DATE NOT NULL,
    window_end_1    DATE NOT NULL,
    window_start_2  DATE NOT NULL,
    window_end_2    DATE NOT NULL,
    similarity      DOUBLE,                 -- How similar are these two periods?
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (window_start_1, window_end_1, window_start_2, window_end_2)
);


-- -----------------------------------------------------------------------------
-- BINDING SCHEMA: Final attachment / positioning
-- -----------------------------------------------------------------------------
-- Indicators positioned within the discovered structure

CREATE TABLE IF NOT EXISTS binding.indicator_positions (
    indicator_id    VARCHAR NOT NULL,
    date            DATE NOT NULL,
    x               DOUBLE,                 -- Position in geometry space
    y               DOUBLE,
    z               DOUBLE,                 -- Optional 3rd dimension
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_id, date)
);

CREATE TABLE IF NOT EXISTS binding.leader_laggard (
    indicator_id    VARCHAR NOT NULL,
    window_start    DATE NOT NULL,
    window_end      DATE NOT NULL,
    role            VARCHAR NOT NULL,       -- 'leader', 'midpack', 'laggard'
    score           DOUBLE,
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_id, window_start, window_end)
);

CREATE TABLE IF NOT EXISTS binding.regime_membership (
    indicator_id    VARCHAR NOT NULL,
    date            DATE NOT NULL,
    regime_id       VARCHAR NOT NULL,
    probability     DOUBLE,                 -- How strongly in this regime?
    run_id          VARCHAR NOT NULL,

    PRIMARY KEY (indicator_id, date, regime_id)
);


-- =============================================================================
-- END SCHEMA
-- =============================================================================
