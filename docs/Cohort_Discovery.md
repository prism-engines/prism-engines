## Cohort Discovery

### Input
- Cleaned wide-panel data
- Audit statistics from data quality agents

### Features Used (per indicator)
Cohorts are discovered using structural features such as:

- `missing_rate`
- `max_gap_ratio`
- `median_dt_days`
- `staleness_rate`
- `coverage_ratio`
- `start_position`
- `end_position`

These features describe **how the data exists**, not what it represents.

### Output
- Cohort assignments
- Cohort-level confidence
- Structural overlap metrics
- Cadence classification

No data mutation occurs during cohort discovery.

---

## Geometry Semantics

Each cohort produces its **own geometry**, meaning:

- Distances are meaningful *within* a cohort
- Eigenstructure is not diluted by sparse indicators
- Engines operate only where their assumptions hold

Multiple geometries may exist simultaneously.

This is not a failure case — it is the expected outcome for heterogeneous data.

---

## Engine Routing

After cohort discovery:

- Engines are **allowed**, **downweighted**, or **suppressed** per cohort
- Decisions are based on:
  - Effective sample size
  - Density
  - Structural stability
  - Audit results

Example:
- Cointegration suppressed for sparse equity cohorts
- Entropy allowed for dense rate cohorts
- HMM routed only where cadence is stable

---

## What Cohort Awareness Prevents

Cohort-aware geometry explicitly prevents:

- Geometry dominated by sparse indicators
- Monthly data distorting daily structures
- Silent forward-filling artifacts
- False coupling across incompatible series

PRISM does not “fix” data to preserve geometry.
It **fixes geometry to reflect data reality**.

---

## What This Enables

Cohort-aware geometry enables:

- Honest multiscale structure detection
- Parallel geometries across domains
- Defensible suppression of models
- Transparent audit trails
- Cross-cohort comparison without contamination

This is foundational for:
- Regime detection
- Structural diagnostics
- Cross-domain research
- Publication-quality analysis

---

## Design Philosophy

PRISM assumes:

- Some data is not usable for some questions
- Some geometries should not exist
- Absence of structure is a valid result

Cohort-aware geometry encodes these assumptions directly into the system.

---

## Summary

Cohort-aware geometry is not an optimization.
It is a **correctness requirement**.

By discovering cohorts before computing geometry, PRISM ensures that every geometric result reflects real, observable structure — and nothing else.

This makes PRISM robust, transparent, and suitable for serious research use.
