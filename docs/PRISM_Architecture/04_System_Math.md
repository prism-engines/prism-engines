# PRISM System Math Subsystem — Technical Specification

## Purpose
Quantify **indicator significance within the full system context**.

## Input
- lens_outputs
- indicator_derived
- system composition metadata
- time

## Output
- Indicator contribution scores
- System-normalized influence metrics

## Constraints
- No raw data access
- No fetching
- No lens re-computation
- Operates only on persisted analytical outputs

## Persistence
Outputs are written to system_metrics.