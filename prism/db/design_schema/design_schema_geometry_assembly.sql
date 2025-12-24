-- =============================================================================
-- PRISM Geometry Assembly Schema
-- Phase-1 tables for constructed system geometry
-- =============================================================================

-- System geometry vectors per (run_id, indicator_id, window)
-- Each row is a point in geometry space
CREATE TABLE IF NOT EXISTS phase1.geometry_vectors (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,
    window_start        DATE NOT NULL,
    window_end          DATE NOT NULL,
    
    -- The geometry vector (stored as JSON array)
    vector              TEXT NOT NULL,           -- JSON array of floats
    vector_dim          INTEGER NOT NULL,        -- Dimensionality
    
    -- Component contributions (which engines contributed)
    component_engines   TEXT NOT NULL,           -- Comma-separated engine names
    component_weights   TEXT,                    -- JSON array of weights per engine
    
    -- Quality metadata
    coverage_fraction   REAL NOT NULL,           -- Fraction of engines that contributed
    effective_confidence REAL NOT NULL,          -- Weighted confidence from validation
    
    -- Timestamps
    assembled_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assembler_version   TEXT NOT NULL,
    
    PRIMARY KEY (run_id, indicator_id, window_years, window_start)
);

-- Geometry trajectories (time series of geometry vectors)
CREATE TABLE IF NOT EXISTS phase1.geometry_trajectories (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,
    
    -- Trajectory metadata
    n_points            INTEGER NOT NULL,        -- Number of time points
    start_date          DATE NOT NULL,
    end_date            DATE NOT NULL,
    
    -- Trajectory statistics
    total_distance      REAL,                    -- Sum of step distances
    avg_step_size       REAL,                    -- Average movement per step
    max_step_size       REAL,                    -- Maximum single-step movement
    trajectory_length   REAL,                    -- Path length through space
    displacement        REAL,                    -- Start-to-end distance
    
    -- Assembled from
    assembler_version   TEXT NOT NULL,
    assembled_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (run_id, indicator_id, window_years)
);

-- Coordinate frame definitions (how engines map to geometry dimensions)
CREATE TABLE IF NOT EXISTS phase1.geometry_coordinate_frames (
    run_id              TEXT NOT NULL,
    frame_id            TEXT NOT NULL,           -- e.g., 'default', 'pca_reduced'
    
    -- Dimension mapping
    dimension_index     INTEGER NOT NULL,
    engine              TEXT NOT NULL,
    engine_output_key   TEXT NOT NULL,           -- Which output from the engine
    transform           TEXT,                    -- Optional transform applied
    
    -- Metadata
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (run_id, frame_id, dimension_index)
);

-- Engine-to-geometry mapping (which engine outputs became which vector components)
CREATE TABLE IF NOT EXISTS phase1.geometry_engine_mapping (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,
    
    engine              TEXT NOT NULL,
    output_key          TEXT NOT NULL,
    vector_indices      TEXT NOT NULL,           -- JSON array of indices this maps to
    contribution_weight REAL NOT NULL,
    
    PRIMARY KEY (run_id, indicator_id, window_years, engine, output_key)
);

-- View: Geometry vectors with trajectory context
CREATE VIEW IF NOT EXISTS phase1.v_geometry_with_trajectory AS
SELECT 
    gv.*,
    gt.n_points,
    gt.total_distance,
    gt.avg_step_size,
    gt.displacement
FROM phase1.geometry_vectors gv
LEFT JOIN phase1.geometry_trajectories gt
    ON gv.run_id = gt.run_id
    AND gv.indicator_id = gt.indicator_id
    AND gv.window_years = gt.window_years;

-- View: Indicators with complete geometry
CREATE VIEW IF NOT EXISTS phase1.v_indicators_with_geometry AS
SELECT DISTINCT
    run_id,
    indicator_id,
    window_years,
    COUNT(*) as n_geometry_points,
    AVG(effective_confidence) as avg_confidence,
    MIN(window_start) as earliest_point,
    MAX(window_end) as latest_point
FROM phase1.geometry_vectors
GROUP BY run_id, indicator_id, window_years;
