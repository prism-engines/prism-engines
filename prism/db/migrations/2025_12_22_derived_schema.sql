-- 2025_12_22_derived_schema.sql
-- Derived Phase Schema: Engine output tables for descriptor-first orchestration
--
-- Tables:
--   derived.geometry_descriptors - Single-indicator descriptors
--   derived.pairwise_descriptors - Pairwise descriptors (MI, granger, etc.)
--   derived.cohort_descriptors - Cohort-level descriptors (PCA)
--   derived.regime_assignments - HMM regime assignments
--   derived.wavelet_coefficients - Wavelet coefficient storage
--   derived.correlation_matrix - Correlation matrix storage
--   meta.engine_runs - Engine run metadata
--   meta.engine_skips - Skipped engines with reasons
--   meta.cohort_membership - Cohort membership


-- Single-indicator descriptors (geometry, entropy, wavelets, etc.)
CREATE TABLE derived.geometry_descriptors (
    indicator_id VARCHAR NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    dimension VARCHAR NOT NULL,
    value DOUBLE,
    run_id VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicator_id, window_start, window_end, dimension, run_id)
);

-- Pairwise descriptors (mutual information, granger, etc.)
CREATE TABLE derived.pairwise_descriptors (
    indicator_id_1 VARCHAR NOT NULL,
    indicator_id_2 VARCHAR NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    dimension VARCHAR NOT NULL,
    value DOUBLE,
    p_value DOUBLE,
    run_id VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicator_id_1, indicator_id_2, window_start, window_end, dimension, run_id)
);

-- Cohort-level descriptors (PCA on indicator groups)
CREATE TABLE derived.cohort_descriptors (
    cohort_id VARCHAR NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    dimension VARCHAR NOT NULL,
    value DOUBLE,
    run_id VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cohort_id, window_start, window_end, dimension, run_id)
);

-- Regime assignments (HMM states)
CREATE TABLE derived.regime_assignments (
    indicator_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    regime INTEGER NOT NULL,
    probability DOUBLE,
    run_id VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicator_id, date, run_id)
);

-- Wavelet coefficients (multi-scale decomposition)
CREATE TABLE derived.wavelet_coefficients (
    indicator_id VARCHAR NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    scale INTEGER NOT NULL,
    coefficient_type VARCHAR NOT NULL,
    value DOUBLE,
    run_id VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicator_id, window_start, window_end, scale, coefficient_type, run_id)
);

-- Correlation matrix storage
CREATE TABLE derived.correlation_matrix (
    indicator_id_1 VARCHAR NOT NULL,
    indicator_id_2 VARCHAR NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    correlation DOUBLE NOT NULL,
    run_id VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicator_id_1, indicator_id_2, window_start, window_end, run_id)
);

-- Engine run metadata
CREATE TABLE meta.engine_runs (
    run_id VARCHAR PRIMARY KEY,
    engine_name VARCHAR NOT NULL,
    phase VARCHAR DEFAULT 'derived',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR DEFAULT 'running',
    rows_written INTEGER DEFAULT 0,
    rows_skipped INTEGER DEFAULT 0,
    window_start DATE,
    window_end DATE,
    normalization VARCHAR,
    parameters VARCHAR,
    error_message VARCHAR
);

-- Skipped engines with reasons
CREATE TABLE meta.engine_skips (
    run_id VARCHAR NOT NULL,
    engine VARCHAR NOT NULL,
    indicator_id VARCHAR NOT NULL,
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    skip_reason VARCHAR NOT NULL,
    skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, engine, indicator_id, window_start, window_end)
);

-- Cohort membership
CREATE TABLE meta.cohort_membership (
    run_id VARCHAR NOT NULL,
    cohort_id VARCHAR NOT NULL,
    indicator_id VARCHAR NOT NULL,
    correlation_to_centroid DOUBLE,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, cohort_id, indicator_id)
);
