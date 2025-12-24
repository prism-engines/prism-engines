-- 2025_12_21_phase1_engine_contract.sql
-- Phase 1 Engine Contract Tables: Atomic, Enforceable, Immutable
--
-- DESIGN INTENT:
-- Phase 1 is an atomic operation.
-- Analysis without persistence is incomplete.
-- Engines must never infer geometry.
-- This is not stylistic. It is architectural.


-- 1) phase1_indicator_windows: Full geometric surface (research/audit layer)
-- Every (indicator, window) pair analyzed during Phase 1
CREATE TABLE meta.phase1_indicator_windows (
    run_id            VARCHAR NOT NULL,
    indicator_id      VARCHAR NOT NULL,
    window_years      DOUBLE  NOT NULL,

    geom_class        VARCHAR NOT NULL,
    quality_score     DOUBLE  NOT NULL,
    confidence        DOUBLE  NOT NULL,
    disagreement      DOUBLE  NOT NULL,
    stability         DOUBLE  NOT NULL,

    PRIMARY KEY (run_id, indicator_id, window_years)
);

-- 2) phase1_indicator_optimal: Canonical engine contract (one row per indicator)
-- The authoritative window/geometry decision engines must use
CREATE TABLE meta.phase1_indicator_optimal (
    run_id            VARCHAR NOT NULL,
    indicator_id      VARCHAR NOT NULL,

    optimal_window    DOUBLE  NOT NULL,
    geom_class        VARCHAR NOT NULL,

    quality_score     DOUBLE  NOT NULL,
    confidence        DOUBLE  NOT NULL,
    disagreement      DOUBLE  NOT NULL,
    stability         DOUBLE  NOT NULL,

    PRIMARY KEY (run_id, indicator_id)
);

-- 3) phase1_math_eligibility: Explicit math gating
-- Engines read this to determine what analysis is permitted
CREATE TABLE meta.phase1_math_eligibility (
    run_id            VARCHAR NOT NULL,
    indicator_id      VARCHAR NOT NULL,

    eligibility       VARCHAR NOT NULL,   -- eligible | conditional | ineligible
    reason            VARCHAR,

    PRIMARY KEY (run_id, indicator_id)
);

-- Note: meta.data_run_lock already exists and is used for enforcement
-- Engines MUST check for lock before execution
