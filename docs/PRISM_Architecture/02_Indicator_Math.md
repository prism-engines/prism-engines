# PRISM Indicator Math Subsystem — Technical Specification

## Purpose
Compute **per-indicator analytical measurements** from raw time-series data.

## Technologies
- Python
- pandas
- NumPy (optional, internal)

## Input
- indicator_values from DuckDB
- Single indicator_id
- Explicit parameters
- Time windows

## Output
- pandas DataFrame
- Indexed by:
  - indicator_id
  - time
  - window_id (if applicable)

## Responsibilities
- Normalization
- Trend extraction
- Rolling statistics
- Indicator-local geometry

## Constraints
- Operates on one indicator at a time
- No cross-indicator assumptions
- No DuckDB access
- No SQL execution
- No persistence decisions

## Persistence
Outputs are written by the runner to indicator_derived.