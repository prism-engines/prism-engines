-- 2025_12_20_phase1_views_and_temporal_obs.sql

CREATE OR REPLACE VIEW meta.v_phase1_steps AS
SELECT
  run_id,
  step_name,
  status,
  started_at,
  completed_at,
  n_items AS items_processed,
  error_json
FROM meta.phase1_steps;

CREATE OR REPLACE VIEW meta.v_geometry_windows AS
SELECT
  run_id,
  indicator_id,
  optimal_window_y AS window_years,
  window_days,
  dominant_geometry,
  geometry_pct,
  avg_confidence,
  avg_disagreement,
  confidence_std,
  n_observations,
  n_transitions,
  stability,
  quality_score,
  created_at
FROM meta.geometry_windows;

CREATE OR REPLACE VIEW meta.v_phase1_reports AS
SELECT
  run_id,
  report_path,
  sha256,
  created_at AS generated_at,
  report_markdown
FROM meta.phase1_reports;

CREATE TABLE meta.temporal_geometry_observations (
  run_id        VARCHAR,
  indicator_id  VARCHAR,
  window_years  DOUBLE,
  window_end    DATE,
  geometry      VARCHAR,
  confidence    DOUBLE,
  disagreement  DOUBLE,
  created_at    TIMESTAMP DEFAULT now(),
  PRIMARY KEY (run_id, indicator_id, window_years, window_end)
);
