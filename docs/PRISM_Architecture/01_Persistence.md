# PRISM Persistence Subsystem — Canonical Technical Specification

## Purpose
The Persistence Subsystem is responsible for **durable state, lineage, and replayability**.
It is the only subsystem permitted to interact directly with DuckDB and execute SQL.

## Technologies
- DuckDB (embedded OLAP database)
- SQL (DuckDB dialect)
- Python (duckdb, pandas)

## Responsibilities
- Persist all fetched, derived, and computed data
- Persist execution metadata and failures
- Provide reproducible historical state
- Enforce schema and type consistency

## Forbidden Responsibilities
- No fetching
- No mathematical computation
- No interpretation or domain logic
- No orchestration decisions

## Core Tables (Logical)
- fetch_runs
- indicator_values
- indicator_derived
- lens_outputs
- system_metrics

## DuckDB Usage
- Single-writer model per run
- SQL used for:
  - time alignment
  - joins
  - filtering
  - window materialization
- No SQL logic embedded in math engines

## Lineage
Every table must include:
- run_id
- indicator_id (where applicable)
- parameter_hash
- execution_timestamp

Persistence stores facts only.