# PRISM Data Management & Lineage — Technical Specification

## Purpose
Ensure traceability, reproducibility, and auditability.

## Lineage Fields (Required)
- run_id
- indicator_id
- source
- parameter_hash
- execution_timestamp

## SQL Role
- Join datasets across time
- Align indicators
- Materialize windows
- Preserve historical state

## DuckDB Role
- Embedded analytical store
- Deterministic replay
- Local + cloud portability

## pandas Role
- In-memory transformation
- Math execution substrate

All transitions between SQL and pandas are explicit.