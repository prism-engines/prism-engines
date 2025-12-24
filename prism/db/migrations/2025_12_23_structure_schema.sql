-- PRISM v3: Structure Schema
-- System geometry and indicator positioning tables

-- System-level geometry metadata
CREATE TABLE structure.system_geometry (
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    n_indicators INTEGER,
    effective_dimension DOUBLE,
    n_cohorts INTEGER,
    mean_correlation DOUBLE,
    network_density DOUBLE,
    geometric_stability DOUBLE,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (window_start, window_end)
);

-- Individual indicator positions within system geometry
CREATE TABLE structure.indicator_positions (
    indicator_id VARCHAR NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    centroid_distance DOUBLE,
    pc1_loading DOUBLE,
    pc2_loading DOUBLE,
    pc3_loading DOUBLE,
    isolation_score DOUBLE,
    degrees_of_freedom DOUBLE,
    structural_centrality DOUBLE,
    systemic_importance DOUBLE,
    information_value DOUBLE,
    geometric_fit DOUBLE,
    anomaly_potential DOUBLE,
    PRIMARY KEY (indicator_id, window_start, window_end)
);
