-- 2025_12_20_phase1_reports.sql
-- Phase 1 report storage: Stores generated reports for audit and reproducibility
--
-- Required by: meta.v_phase1_reports view in 2025_12_20_phase1_views_and_temporal_obs.sql


CREATE TABLE meta.phase1_reports (
    run_id           VARCHAR NOT NULL,
    report_path      VARCHAR NOT NULL,
    sha256           VARCHAR,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    report_markdown  VARCHAR,
    PRIMARY KEY (run_id, report_path)
);
