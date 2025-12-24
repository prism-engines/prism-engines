-- =============================================================================
-- PRISM Engine Eligibility Schema v2 (Window-Aware)
-- =============================================================================
--
-- Eligibility is PER-INDICATOR-PER-WINDOW, not just per-indicator.
-- Different math needs different temporal resolution.
--
-- Flow:
--   Phase 1:   Temporal Geometry Scan → meta.geometry_windows
--   Phase 1.5: Math Suitability Agent → meta.engine_eligibility
--   Phase 2:   Engine Runner queries eligibility, runs only certified combinations
--
-- =============================================================================

-- -----------------------------------------------------------------------------
-- GEOMETRY WINDOWS TABLE
-- -----------------------------------------------------------------------------
-- Stores geometry signatures at each tested window for each indicator.
-- This is the output of the Temporal Geometry Scan.

CREATE TABLE IF NOT EXISTS meta.geometry_windows (
    run_id              VARCHAR NOT NULL,
    indicator_id        VARCHAR NOT NULL,
    window_years        DOUBLE NOT NULL,       -- 0.5, 1.0, 2.0, 3.0, 5.0, 7.0
    window_days         INTEGER NOT NULL,      -- Actual trading days
    
    -- Geometry at this window
    dominant_geometry   VARCHAR,               -- Most frequent geometry
    geometry_pct        DOUBLE,                -- Percentage of time in dominant
    
    -- Quality metrics
    avg_confidence      DOUBLE,
    avg_disagreement    DOUBLE,
    confidence_std      DOUBLE,
    
    -- Stability
    n_observations      INTEGER,
    n_transitions       INTEGER,
    stability           DOUBLE,                -- 1.0 = no transitions
    
    -- Combined quality score
    quality_score       DOUBLE,
    
    -- Is this the optimal window for this indicator?
    is_optimal          BOOLEAN DEFAULT FALSE,
    
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (run_id, indicator_id, window_years)
);


-- -----------------------------------------------------------------------------
-- ENGINE ELIGIBILITY TABLE (Window-Aware)
-- -----------------------------------------------------------------------------
-- Stores engine routing decisions PER INDICATOR PER WINDOW.
-- An indicator may be eligible at 3-year but ineligible at 1-year.

CREATE TABLE IF NOT EXISTS meta.engine_eligibility (
    run_id                VARCHAR NOT NULL,
    indicator_id          VARCHAR NOT NULL,
    window_years          DOUBLE NOT NULL,     -- Which window this applies to
    
    -- Geometry context at this window
    geometry              VARCHAR,             -- Geometry at this window
    confidence            DOUBLE,
    disagreement          DOUBLE,
    stability             DOUBLE,
    
    -- Suitability decision
    status                VARCHAR NOT NULL,    -- eligible | conditional | ineligible | unknown
    
    -- Engine routing for THIS window
    allowed_engines       JSON,
    prohibited_engines    JSON,
    conditional_engines   JSON,
    
    -- Epistemic trace
    rationale             JSON,
    policy_version        VARCHAR NOT NULL,
    
    created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (run_id, indicator_id, window_years)
);


-- -----------------------------------------------------------------------------
-- VIEW-LEVEL GEOMETRY (per window)
-- -----------------------------------------------------------------------------
-- Optional detailed view for each (indicator, window, view) combination.

CREATE TABLE IF NOT EXISTS meta.geometry_window_views (
    run_id            VARCHAR NOT NULL,
    indicator_id      VARCHAR NOT NULL,
    window_years      DOUBLE NOT NULL,
    view              VARCHAR NOT NULL,        -- level | returns | volatility | deviation
    
    geometry          VARCHAR,
    confidence        DOUBLE,
    
    PRIMARY KEY (run_id, indicator_id, window_years, view)
);


-- -----------------------------------------------------------------------------
-- INDEXES
-- -----------------------------------------------------------------------------

-- Find optimal windows
CREATE INDEX IF NOT EXISTS idx_geometry_windows_optimal 
ON meta.geometry_windows(run_id, indicator_id, is_optimal);

-- Find eligible combinations
CREATE INDEX IF NOT EXISTS idx_eligibility_status 
ON meta.engine_eligibility(run_id, status);

-- Query by indicator
CREATE INDEX IF NOT EXISTS idx_eligibility_indicator 
ON meta.engine_eligibility(run_id, indicator_id);

-- Query by window
CREATE INDEX IF NOT EXISTS idx_eligibility_window 
ON meta.engine_eligibility(run_id, window_years);


-- -----------------------------------------------------------------------------
-- CONVENIENCE VIEW: What can I run?
-- -----------------------------------------------------------------------------
-- Engine runner queries this view to get certified combinations.

CREATE VIEW IF NOT EXISTS meta.v_certified_runs AS
SELECT 
    e.run_id,
    e.indicator_id,
    e.window_years,
    e.geometry,
    e.confidence,
    e.status,
    e.allowed_engines,
    g.quality_score,
    g.is_optimal
FROM meta.engine_eligibility e
JOIN meta.geometry_windows g 
    ON e.run_id = g.run_id 
    AND e.indicator_id = g.indicator_id 
    AND e.window_years = g.window_years
WHERE e.status IN ('eligible', 'conditional');
