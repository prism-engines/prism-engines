# PRISM Runners — Technical Specification

## Purpose
Coordinate execution across subsystems.

## Responsibilities
- Read YAML configuration
- Resolve indicators to sources
- Generate run_id
- Invoke fetchers, math engines
- Invoke persistence

## Constraints
- No math implementation
- No SQL schema definition
- No data interpretation

Runners orchestrate only.