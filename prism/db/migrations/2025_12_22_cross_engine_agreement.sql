-- 2025_12_22_cross_engine_agreement.sql
-- Cross-Engine Agreement Schema: Phase-1 annotations for structural coherence detection
--
-- Tables:
--   phase1.cross_engine_agreement - Agreement scores per (run_id, indicator_id, window_years)
--   phase1.engine_pair_agreement - Pairwise engine comparisons
--   phase1.engine_contradictions - Contradiction flags
-- Views:
--   phase1.v_high_agreement_indicators - Indicators with high agreement
--   phase1.v_contradicting_indicators - Indicators with contradictions


-- Agreement scores per (run_id, indicator_id, window_years)
CREATE TABLE phase1.cross_engine_agreement (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,

    -- Overall agreement
    agreement_score     REAL NOT NULL,           -- [0, 1] overall agreement
    n_engines           INTEGER NOT NULL,        -- Number of engines compared

    -- Reinforcement detection
    n_reinforcing_pairs INTEGER NOT NULL,        -- Pairs that agree
    n_contradicting_pairs INTEGER NOT NULL,      -- Pairs that disagree
    n_neutral_pairs     INTEGER NOT NULL,        -- Pairs with no clear signal

    -- Agreement breakdown
    structural_agreement REAL,                   -- Agreement on structure (PCA, correlation)
    temporal_agreement   REAL,                   -- Agreement on dynamics (HMM, DMD)
    scale_agreement      REAL,                   -- Agreement on scale (wavelet, Hurst)

    -- Confidence adjustment
    confidence_multiplier REAL DEFAULT 1.0,      -- Multiplier based on agreement

    -- Metadata
    analyzed_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analyzer_version    TEXT NOT NULL,

    PRIMARY KEY (run_id, indicator_id, window_years)
);

-- Pairwise engine comparisons
CREATE TABLE phase1.engine_pair_agreement (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,

    engine_a            TEXT NOT NULL,
    engine_b            TEXT NOT NULL,

    -- Agreement metrics
    agreement_type      TEXT NOT NULL,           -- 'reinforcing', 'contradicting', 'neutral'
    agreement_strength  REAL NOT NULL,           -- [0, 1] strength of agreement/disagreement

    -- Correlation between contributions
    correlation         REAL,                    -- Pearson correlation of geometry contributions

    -- Transition agreement
    transition_overlap  REAL,                    -- Fraction of transitions both engines detect

    PRIMARY KEY (run_id, indicator_id, window_years, engine_a, engine_b)
);

-- Contradiction flags (specific disagreements to investigate)
CREATE TABLE phase1.engine_contradictions (
    run_id              TEXT NOT NULL,
    indicator_id        TEXT NOT NULL,
    window_years        REAL NOT NULL,

    contradiction_id    INTEGER NOT NULL,

    engine_a            TEXT NOT NULL,
    engine_b            TEXT NOT NULL,

    -- What they disagree about
    dimension           TEXT NOT NULL,           -- Which geometry dimension
    value_a             REAL,
    value_b             REAL,
    disagreement_magnitude REAL NOT NULL,

    -- Impact
    severity            TEXT NOT NULL,           -- 'low', 'medium', 'high'

    PRIMARY KEY (run_id, indicator_id, window_years, contradiction_id)
);

-- View: Indicators with high agreement (reliable structure)
CREATE OR REPLACE VIEW phase1.v_high_agreement_indicators AS
SELECT *
FROM phase1.cross_engine_agreement
WHERE agreement_score >= 0.6
  AND n_contradicting_pairs = 0
ORDER BY agreement_score DESC;

-- View: Indicators with contradictions (need investigation)
CREATE OR REPLACE VIEW phase1.v_contradicting_indicators AS
SELECT
    cea.run_id,
    cea.indicator_id,
    cea.window_years,
    cea.agreement_score,
    cea.n_contradicting_pairs,
    COUNT(ec.contradiction_id) as n_contradictions,
    MAX(ec.disagreement_magnitude) as max_disagreement
FROM phase1.cross_engine_agreement cea
LEFT JOIN phase1.engine_contradictions ec
    ON cea.run_id = ec.run_id
    AND cea.indicator_id = ec.indicator_id
    AND cea.window_years = ec.window_years
WHERE cea.n_contradicting_pairs > 0
GROUP BY cea.run_id, cea.indicator_id, cea.window_years,
         cea.agreement_score, cea.n_contradicting_pairs;
