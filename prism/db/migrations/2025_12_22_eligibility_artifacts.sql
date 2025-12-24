-- 2025_12_22_eligibility_artifacts.sql
-- Additional eligibility artifacts: views and indexes for engine routing
--
-- Note: Core eligibility tables (meta.geometry_windows, meta.engine_eligibility)
-- already exist in 2025_12_20_phase1_artifacts.sql
--
-- This migration adds:
--   meta.geometry_window_views - Optional detailed view per (indicator, window, view)
--   meta.v_certified_runs - Convenience view for certified combinations
--   Indexes for common queries


-- Optional detailed view for each (indicator, window, view) combination
CREATE TABLE meta.geometry_window_views (
    run_id            VARCHAR NOT NULL,
    indicator_id      VARCHAR NOT NULL,
    window_years      DOUBLE NOT NULL,
    view              VARCHAR NOT NULL,        -- level | returns | volatility | deviation

    geometry          VARCHAR,
    confidence        DOUBLE,

    PRIMARY KEY (run_id, indicator_id, window_years, view)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_geometry_windows_optimal
ON meta.geometry_windows(run_id, indicator_id);

CREATE INDEX IF NOT EXISTS idx_eligibility_status
ON meta.engine_eligibility(run_id, status);

CREATE INDEX IF NOT EXISTS idx_eligibility_indicator
ON meta.engine_eligibility(run_id, indicator_id);

CREATE INDEX IF NOT EXISTS idx_eligibility_window
ON meta.engine_eligibility(run_id, window_years);

-- Convenience view: What can I run?
-- Engine runner queries this view to get certified combinations
CREATE OR REPLACE VIEW meta.v_certified_runs AS
SELECT
    e.run_id,
    e.indicator_id,
    e.window_years,
    e.geometry,
    e.confidence,
    e.status,
    e.allowed_engines,
    g.quality_score
FROM meta.engine_eligibility e
JOIN meta.geometry_windows g
    ON e.run_id = g.run_id
    AND e.indicator_id = g.indicator_id
WHERE e.status IN ('eligible', 'conditional');
