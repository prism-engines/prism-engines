-- 2025_12_21_canonical_data_schema.sql
-- Canonical Data Schema: Single source of truth for indicator data
--
-- CANONICAL PHASE MODEL:
--   data      - fetched + cleaned + admissible indicator data
--   derived   - first-order engine measurements (metrics of indicators)
--   structure - geometric organization (coherence, persistence, agreement)
--   binding   - final attachment / positioning within structure
--   meta      - orchestration, audits, windows
--
-- Legacy schemas (raw, clean, unbound, bounded) are deprecated.
-- All new code should use the canonical schemas.


-- Primary indicator data table
-- Combines fetch + clean into single canonical source
CREATE TABLE data.indicators (
    indicator_id      VARCHAR NOT NULL,
    date              DATE NOT NULL,
    value             DOUBLE NOT NULL,
    source            VARCHAR,           -- fred, tiingo, etc.
    frequency         VARCHAR,           -- daily, weekly, monthly
    cleaning_method   VARCHAR,           -- interpolation method used
    fetched_at        TIMESTAMP,
    cleaned_at        TIMESTAMP,
    run_id            VARCHAR,
    PRIMARY KEY (indicator_id, date)
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_data_indicators_id ON data.indicators(indicator_id);
CREATE INDEX IF NOT EXISTS idx_data_indicators_date ON data.indicators(date);

-- View for backward compatibility with clean.indicator_values
CREATE OR REPLACE VIEW clean.indicator_values_v2 AS
SELECT indicator_id, date, value
FROM data.indicators;

-- Note: Derived phase tables will be created separately
-- CREATE SCHEMA IF NOT EXISTS derived;
-- CREATE SCHEMA IF NOT EXISTS binding;
