-- 2025_12_22_temporal_persistence.sql
-- Temporal Persistence Schema: Phase-1 annotations for geometry continuity and stability
--
-- Tables:
--   phase1.temporal_persistence - Persistence metrics per (run_id, indicator_id, window_years)
--   phase1.detected_transitions - Detected transitions (regime boundaries)
--   phase1.window_continuity - Window-to-window continuity details
-- Views:
--   phase1.v_persistent_indicators - Indicators with high persistence
--   phase1.v_transitions_with_context - All transitions with context


-- Persistence metrics per (run_id, indicator_id, window_years)
CREATE TABLE phase1.temporal_persistence (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,

    -- Overall persistence scores
    persistence_score   REAL NOT NULL,           -- [0, 1] overall persistence
    stability_score     REAL NOT NULL,           -- [0, 1] motion stability
    continuity_score    REAL NOT NULL,           -- [0, 1] window-to-window continuity

    -- Transition analysis
    n_transitions       INTEGER NOT NULL,        -- Number of detected transitions
    transition_rate     REAL,                    -- Transitions per time unit
    avg_transition_magnitude REAL,               -- Average size of transitions
    max_transition_magnitude REAL,               -- Largest single transition

    -- Regime persistence
    avg_regime_duration REAL,                    -- Average time in same regime
    max_regime_duration REAL,                    -- Longest regime duration
    regime_entropy      REAL,                    -- Entropy of regime time distribution

    -- Boundary artifacts
    edge_instability    REAL,                    -- Instability at window edges
    boundary_artifact_flag BOOLEAN DEFAULT FALSE,

    -- Metadata
    analyzed_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analyzer_version    TEXT NOT NULL,

    PRIMARY KEY (run_id, indicator_id, window_years)
);

-- Detected transitions (regime boundaries)
CREATE TABLE phase1.detected_transitions (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,

    transition_id       INTEGER NOT NULL,        -- Sequential ID within indicator
    transition_date     DATE NOT NULL,

    -- Transition characteristics
    magnitude           REAL NOT NULL,           -- Distance in geometry space
    direction_vector    TEXT,                    -- JSON array of direction components
    sharpness           REAL,                    -- How abrupt (higher = sharper)
    confidence          REAL NOT NULL,           -- Confidence in transition detection

    -- Pre/post states
    pre_state_centroid  TEXT,                    -- JSON array
    post_state_centroid TEXT,                    -- JSON array

    PRIMARY KEY (run_id, indicator_id, window_years, transition_id)
);

-- Window-to-window continuity details
CREATE TABLE phase1.window_continuity (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,

    window_start        DATE NOT NULL,
    window_end          DATE NOT NULL,
    next_window_start   DATE,

    -- Continuity metrics
    step_distance       REAL NOT NULL,           -- Distance to next point
    velocity            REAL,                    -- Distance / time
    acceleration        REAL,                    -- Change in velocity

    -- Smoothness
    smoothness_score    REAL,                    -- [0, 1] how smooth the transition
    is_discontinuity    BOOLEAN DEFAULT FALSE,   -- Flagged as discontinuous

    PRIMARY KEY (run_id, indicator_id, window_years, window_start)
);

-- View: Indicators with high persistence
CREATE OR REPLACE VIEW phase1.v_persistent_indicators AS
SELECT *
FROM phase1.temporal_persistence
WHERE persistence_score >= 0.5
  AND boundary_artifact_flag = FALSE
ORDER BY persistence_score DESC;

-- View: All transitions with context
CREATE OR REPLACE VIEW phase1.v_transitions_with_context AS
SELECT
    dt.*,
    tp.persistence_score,
    tp.stability_score,
    tp.avg_transition_magnitude
FROM phase1.detected_transitions dt
JOIN phase1.temporal_persistence tp
    ON dt.run_id = tp.run_id
    AND dt.indicator_id = tp.indicator_id
    AND dt.window_years = tp.window_years;
