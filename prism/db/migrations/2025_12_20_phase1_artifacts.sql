-- 2025_12_20_phase1_artifacts.sql
-- Phase 1 persistence tables: window selection + suitability + run logs + per-window trace


-- 1) Phase 1 run header
CREATE TABLE meta.phase1_runs (
  run_id                VARCHAR PRIMARY KEY,
  domain                VARCHAR,
  mode                  VARCHAR,
  started_at            TIMESTAMP,
  completed_at          TIMESTAMP,
  indicators_requested  INTEGER,
  indicators_scanned    INTEGER,
  windows_json          VARCHAR,
  policy_version        VARCHAR,
  notes                 VARCHAR
);

-- 2) Step-level logs (so failures are queryable)
CREATE TABLE meta.phase1_steps (
  run_id       VARCHAR,
  step_name    VARCHAR,
  status       VARCHAR,          -- success|failed|skipped
  started_at   TIMESTAMP,
  completed_at TIMESTAMP,
  n_items      INTEGER,
  error_json   VARCHAR,
  PRIMARY KEY (run_id, step_name)
);

-- 3) Optimal windows per indicator (the missing table causing your crash)
-- Note: geometry_windows may already exist - this is CREATE IF NOT EXISTS
CREATE TABLE meta.geometry_windows (
  run_id            VARCHAR,
  indicator_id      VARCHAR,
  optimal_window_y  DOUBLE,
  window_days       INTEGER,
  dominant_geometry VARCHAR,
  geometry_pct      DOUBLE,
  avg_confidence    DOUBLE,
  avg_disagreement  DOUBLE,
  confidence_std    DOUBLE,
  n_observations    INTEGER,
  n_transitions     INTEGER,
  stability         DOUBLE,
  quality_score     DOUBLE,
  created_at        TIMESTAMP DEFAULT now(),
  PRIMARY KEY (run_id, indicator_id)
);

-- 4) Suitability (per indicator, per window)
CREATE TABLE meta.engine_eligibility (
  run_id              VARCHAR,
  indicator_id        VARCHAR,
  window_years        DOUBLE,
  geometry            VARCHAR,
  confidence          DOUBLE,
  disagreement        DOUBLE,
  stability           DOUBLE,
  status              VARCHAR,     -- eligible|conditional|ineligible|unknown
  allowed_engines     VARCHAR,     -- JSON list
  prohibited_engines  VARCHAR,     -- JSON list
  conditional_engines VARCHAR,     -- JSON list
  rationale           VARCHAR,     -- JSON list
  policy_version      VARCHAR,
  created_at          TIMESTAMP,
  PRIMARY KEY (run_id, indicator_id, window_years)
);

-- 5) Optional: full scan results (every indicator-window pair)
CREATE TABLE meta.phase1_scan_results (
  run_id            VARCHAR,
  indicator_id      VARCHAR,
  window_years      DOUBLE,
  window_days       INTEGER,
  dominant_geometry VARCHAR,
  geometry_pct      DOUBLE,
  avg_confidence    DOUBLE,
  avg_disagreement  DOUBLE,
  confidence_std    DOUBLE,
  n_observations    INTEGER,
  n_transitions     INTEGER,
  stability         DOUBLE,
  quality_score     DOUBLE,
  is_optimal        BOOLEAN,
  created_at        TIMESTAMP DEFAULT now(),
  PRIMARY KEY (run_id, indicator_id, window_years)
);
