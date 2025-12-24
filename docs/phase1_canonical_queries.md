# Phase 1 Canonical Queries (DuckDB)

## Open the right DB

If you see "Connected to a transient in-memory database", you are NOT on the PRISM DB.

```bash
duckdb data/prism.duckdb
```

Or inside DuckDB:

```sql
.open data/prism.duckdb
```

---

## Runs

```sql
SELECT run_id, domain, mode, started_at, completed_at, policy_version, notes
FROM meta.phase1_runs
ORDER BY started_at DESC
LIMIT 20;
```

---

## Steps (stable view)

```sql
SELECT run_id, step_name, status, items_processed, started_at, completed_at
FROM meta.v_phase1_steps
WHERE run_id = '<RUN_ID>'
ORDER BY step_name;
```

---

## Windows (stable view)

```sql
SELECT *
FROM meta.v_geometry_windows
WHERE run_id = '<RUN_ID>'
ORDER BY indicator_id;
```

---

## Eligibility

```sql
SELECT indicator_id, window_years, status, geometry, confidence, disagreement, stability, policy_version
FROM meta.engine_eligibility
WHERE run_id = '<RUN_ID>'
ORDER BY indicator_id, window_years;
```

---

## Reports (stable view)

```sql
SELECT run_id, report_path, generated_at, sha256
FROM meta.v_phase1_reports
ORDER BY generated_at DESC
LIMIT 20;
```

---

## Temporal observations (if enabled)

```sql
SELECT window_end, geometry, confidence, disagreement
FROM meta.temporal_geometry_observations
WHERE run_id = '<RUN_ID>' AND indicator_id = 'SPY' AND window_years = 3.0
ORDER BY window_end;
```
