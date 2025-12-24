# PRISM Lens Math Subsystem — Technical Specification

## Purpose
Apply **shared mathematical engines** across multiple indicators to measure
relationships and geometry.

## Key Constraint
Indicator math engines and lens math engines **share the same implementations**.

## Input
- indicator_derived datasets
- Time-aligned via SQL
- Explicit lens parameters

## Output
- pandas DataFrame
- Cross-indicator metrics
- Geometric descriptors

## Constraints
- No fetching
- No raw data access
- No persistence logic
- No domain interpretation

## Persistence
Lens outputs are written to lens_outputs.