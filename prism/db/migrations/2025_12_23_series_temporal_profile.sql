-- 2025_12_23_series_temporal_profile.sql
-- Native Sampling Infrastructure: Temporal profiles for indicators
--
-- PURPOSE:
--   Enable native sampling by storing computed temporal characteristics
--   for each indicator. Engines can use this to:
--   1. Validate they have enough observations for meaningful analysis
--   2. Interpret time-series correctly (daily vs monthly vs quarterly)
--   3. Align pairwise/cohort analysis appropriately
--
-- NATIVE SAMPLING PRINCIPLE:
--   "Operate on native series at native cadence without resampling"
--   Data integrity > convenience. Missing = NULL, not interpolated.

-- Temporal profile per indicator
-- Computed from actual data intervals, not declared frequency
CREATE TABLE meta.series_temporal_profile (
    indicator_id        VARCHAR PRIMARY KEY,

    -- Inferred frequency classification
    native_frequency    VARCHAR NOT NULL,       -- 'daily', 'weekly', 'monthly', 'quarterly', 'annual', 'irregular'

    -- Actual interval statistics (in days)
    median_interval_days DOUBLE NOT NULL,       -- Median gap between observations
    min_interval_days   DOUBLE NOT NULL,        -- Minimum gap (detect irregularities)
    max_interval_days   DOUBLE NOT NULL,        -- Maximum gap (detect missing periods)
    stddev_interval_days DOUBLE,                -- Standard deviation of intervals

    -- Regularity assessment
    is_irregular        BOOLEAN DEFAULT FALSE,  -- True if intervals vary significantly
    regularity_score    DOUBLE,                 -- [0, 1] how regular the series is

    -- Coverage statistics
    total_observations  INTEGER NOT NULL,
    first_date          DATE NOT NULL,
    last_date           DATE NOT NULL,
    coverage_days       INTEGER NOT NULL,       -- last_date - first_date
    expected_obs        INTEGER,                -- Expected observations given frequency
    coverage_ratio      DOUBLE,                 -- actual / expected

    -- Metadata
    inferred_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    inference_version   VARCHAR DEFAULT '1.0.0'
);

-- Index for frequency-based queries
CREATE INDEX idx_temporal_profile_freq
    ON meta.series_temporal_profile(native_frequency);

-- View: Indicators suitable for daily analysis
CREATE OR REPLACE VIEW meta.v_daily_indicators AS
SELECT indicator_id, total_observations, regularity_score
FROM meta.series_temporal_profile
WHERE native_frequency = 'daily'
  AND regularity_score >= 0.8
ORDER BY total_observations DESC;

-- View: Indicators suitable for monthly analysis
CREATE OR REPLACE VIEW meta.v_monthly_indicators AS
SELECT indicator_id, total_observations, regularity_score
FROM meta.series_temporal_profile
WHERE native_frequency IN ('monthly', 'quarterly')
  AND regularity_score >= 0.7
ORDER BY total_observations DESC;
