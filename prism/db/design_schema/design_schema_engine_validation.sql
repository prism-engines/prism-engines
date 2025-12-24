-- =============================================================================
-- PRISM Engine Output Validation Schema
-- Phase-1 annotations for engine output quality control
-- =============================================================================

-- Validation results per (run_id, indicator_id, window_years, engine)
CREATE TABLE IF NOT EXISTS phase1.engine_validation (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,
    engine              TEXT NOT NULL,
    
    -- Validation outcome
    validity_flag       TEXT NOT NULL,          -- 'valid', 'degraded', 'invalid'
    confidence_penalty  REAL DEFAULT 0.0,       -- [0, 1] penalty to apply
    
    -- Reason codes (comma-separated)
    reason_codes        TEXT,                   -- e.g., 'FLAT_EIGENVALUES,LOW_ENTROPY'
    
    -- Detailed metrics
    numerical_stability REAL,                   -- [0, 1] stability score
    information_content REAL,                   -- [0, 1] information score
    degeneracy_score    REAL,                   -- [0, 1] higher = more degenerate
    window_coverage     REAL,                   -- fraction of window with valid data
    
    -- Metadata
    validated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validator_version   TEXT NOT NULL,
    
    PRIMARY KEY (run_id, indicator_id, window_years, engine)
);

-- Reason code definitions
CREATE TABLE IF NOT EXISTS phase1.validation_reason_codes (
    code            TEXT PRIMARY KEY,
    severity        TEXT NOT NULL,              -- 'warning', 'error', 'fatal'
    description     TEXT NOT NULL,
    penalty_weight  REAL DEFAULT 0.1
);

-- Insert standard reason codes
INSERT OR REPLACE INTO phase1.validation_reason_codes VALUES
    ('FLAT_EIGENVALUES', 'error', 'PCA eigenvalues are flat (no dominant structure)', 0.3),
    ('ZERO_VARIANCE', 'fatal', 'Output has zero variance', 1.0),
    ('NAN_VALUES', 'fatal', 'Output contains NaN values', 1.0),
    ('INF_VALUES', 'fatal', 'Output contains infinite values', 1.0),
    ('LOW_ENTROPY', 'warning', 'HMM state entropy below threshold', 0.2),
    ('DEGENERATE_STATES', 'error', 'HMM collapsed to single state', 0.4),
    ('INSUFFICIENT_TRANSITIONS', 'warning', 'Too few state transitions detected', 0.2),
    ('NUMERICAL_OVERFLOW', 'fatal', 'Numerical overflow detected', 1.0),
    ('NUMERICAL_UNDERFLOW', 'warning', 'Numerical underflow detected', 0.1),
    ('WINDOW_SPARSE', 'warning', 'Less than 50% window coverage', 0.2),
    ('WINDOW_MINIMAL', 'error', 'Less than 20% window coverage', 0.5),
    ('CONSTANT_OUTPUT', 'error', 'Engine produced constant output', 0.5),
    ('CORRELATION_DEGENERATE', 'warning', 'Correlation matrix nearly singular', 0.2),
    ('GRANGER_UNSTABLE', 'error', 'Granger causality numerically unstable', 0.4),
    ('WAVELET_EDGE_EFFECTS', 'warning', 'Significant wavelet edge artifacts', 0.15),
    ('DMD_UNSTABLE_MODES', 'warning', 'DMD produced unstable modes', 0.2),
    ('HURST_BOUNDARY', 'warning', 'Hurst exponent at boundary (0 or 1)', 0.15);

-- View for downstream agents: only valid/degraded outputs
CREATE VIEW IF NOT EXISTS phase1.v_validated_outputs AS
SELECT 
    ev.*,
    CASE 
        WHEN ev.validity_flag = 'valid' THEN 1.0 - ev.confidence_penalty
        WHEN ev.validity_flag = 'degraded' THEN 0.5 * (1.0 - ev.confidence_penalty)
        ELSE 0.0
    END AS effective_confidence
FROM phase1.engine_validation ev
WHERE ev.validity_flag IN ('valid', 'degraded');

-- Aggregated validation summary per (indicator, window)
CREATE VIEW IF NOT EXISTS phase1.v_validation_summary AS
SELECT
    run_id,
    indicator_id,
    window_years,
    COUNT(*) AS total_engines,
    SUM(CASE WHEN validity_flag = 'valid' THEN 1 ELSE 0 END) AS valid_count,
    SUM(CASE WHEN validity_flag = 'degraded' THEN 1 ELSE 0 END) AS degraded_count,
    SUM(CASE WHEN validity_flag = 'invalid' THEN 1 ELSE 0 END) AS invalid_count,
    AVG(confidence_penalty) AS avg_penalty,
    AVG(numerical_stability) AS avg_stability,
    AVG(information_content) AS avg_information
FROM phase1.engine_validation
GROUP BY run_id, indicator_id, window_years;
