# PRISM Glossary

> PRISM does not name states, predict events, or classify regimes.  
> It measures system geometry, displacement from historical homeostasis,  
> and the empirical outcomes associated with similar structural configurations.

---

## Core Concepts

### Geometry
The instantaneous structural configuration of the system, as measured by PRISM engines.

### Geometry Vector (G)
Numeric representation of system structure at time t. The output of all contributing engines, concatenated.

### Geometry Signature
The characteristic pattern of engine outputs that defines current structural state.

### State Space
The historical space of all observed geometry vectors. PRISM locates current geometry within this space.

---

## Reference Objects

### Homeostatic Band (HSB)
The region of state space representing "normal" structural variation. Defined by density, not percentile. The system is either inside or outside this band — never "in a regime."

### Reference Geometry
The center (μ) and covariance (Σ) of the homeostatic band. The anchor point for all displacement measurements.

### Band Radius (r)
The coverage-defined threshold. Points with displacement ≤ r are inside the band.

---

## Distance & Deformation

### Displacement (D)
Scalar distance from reference geometry. Computed via Mahalanobis distance or density-based methods. This is the primary PRISM output — not a label, but a measurement.

### Deformation
Sustained or directional displacement from homeostasis. A process, not a state.

### Excursion
A contiguous period spent outside the Homeostatic Band. Defined by entry (crossing out) and exit (crossing back).

### Excursion Depth
Maximum displacement reached during an excursion.

### Excursion Duration
Number of time periods spent outside the band during an excursion.

### Rarity Score
Percentile of current displacement vs historical distribution. "How unusual is this geometry?"

---

## Motion & Dynamics

### Velocity (v)
First derivative of displacement: v(t) = D(t) - D(t-1). Positive = moving away from homeostasis. Negative = reverting.

### Acceleration (a)
Second derivative of displacement: a(t) = v(t) - v(t-1). Positive = deformation compounding. Negative = deformation stabilizing.

### Trajectory
The path of geometry through state space over time. Defined by (D, v, a).

### Trajectory Types
- **Reverting**: Outside band, velocity negative (returning)
- **Escalating**: Outside band, velocity positive (departing further)
- **Stalling**: Outside band, velocity near zero (stuck)
- **Inside**: Within homeostatic band

### Drift
Slow, persistent movement away from homeostasis without sharp acceleration.

### Reversion
Movement back toward the homeostatic band. A direction, not a guarantee.

### Deformation Acceleration
Sustained positive acceleration while outside band. The geometry-native replacement for "crisis."

---

## Time & Frequency Metrics

### Churn
Frequency of band crossings (entries + exits) per unit time. High churn = unstable geometry.

### Dwell Time
Time spent inside or outside the band per episode.

### Persistence
Tendency for excursions to continue once initiated. Measured by autocorrelation of the outside indicator.

### Reversion Half-Life
Expected time to return halfway from current displacement to the band. Computed from mean-reversion coefficient β.

### Time Since Homeostasis
Days since the system was last inside the band.

---

## Probabilistic Outputs

### Conditional Outcome Frequency (COF)
Empirical frequency of specific outcomes following geometrically similar historical episodes.

**Canonical phrasing:**
> "Among the K most similar historical geometries, X% experienced outcome Y within horizon H."

### Escalation Hazard
P(move to higher displacement level | current D, v) within horizon h.

### Reversion Hazard
P(re-enter homeostatic band | current D, v) within horizon h.

### Outcome Horizon
The time window over which conditional outcomes are measured. Standard horizons: 4 weeks, 12 weeks.

### Structural Precedent
Historical episodes with geometry vectors similar to current configuration. The basis for COF estimation.

---

## Mean Reversion & Pull

### Mean-Reversion Coefficient (β)
From model: ΔD(t) = α + β(D(t-1) - r) + ε
- β < 0: System pulled back toward band (mean-reverting)
- β ≈ 0: Random walk (no restoring force)
- β > 0: System tends to diverge (unstable)

### Pull Strength
Magnitude of restoring force at current displacement. |β| when mean-reverting, 0 otherwise.

### Resilience
Qualitative: A system with strong pull strength and short reversion half-life. PRISM does not output "resilient" — it outputs the metrics that define it.

---

## Indicator Attribution

### Attribution
Statistical association between indicator changes and displacement dynamics.

### Activation
Indicators that change materially following displacement events.

### Lead Time
How many periods before an excursion an indicator begins moving.

### Effect Size
Standardized magnitude of indicator change during excursions.

### Sensitivity Axis
Geometry dimensions most responsible for current displacement. Used for explainability.

---

## Governance & Enforcement

### Semantic Guard
Runtime enforcement preventing forbidden tokens (regime, bull, bear, crisis, etc.) from appearing in core outputs.

### Geometry-First Rule
"No output may name a state; only measure position and motion."

### Interpretation Layer
Explicitly outside core execution. Where humans may describe outputs using domain language. PRISM core never interprets.

---

## Forbidden Language (Hard Ban in Core)

These tokens must NEVER appear in PRISM core outputs:

| Forbidden | Replacement |
|-----------|-------------|
| regime | (none — concept removed) |
| bull / bear | (none) |
| crisis | excursion, deformation acceleration |
| risk-on / risk-off | (none) |
| stressed / calm | displaced / inside band |
| normal state | inside homeostatic band |
| state name (any) | state_id (integer only) |

---

## Signature PRISM Phrases

Use these in documentation and communication:

- "The system is displaced, not classified."
- "PRISM locates the system relative to its own history."
- "Stress is distance, not a label."
- "Outcomes are conditioned on geometry, not regimes."
- "PRISM measures motion through structure."
- "Where are we, how fast are we moving, and what usually happens next?"

---

## PRISM Output Contract

Every PRISM run produces:

1. **Displacement** (D) — distance from homeostasis
2. **Trajectory** (v, a) — velocity and acceleration
3. **Excursion Metrics** — streak length, churn, dwell time
4. **Hazards** — P(reversion), P(escalation) over horizons
5. **COF** — conditional outcome frequencies from structural precedent
6. **Attribution** — which indicators lead displacement dynamics
7. **Confidence** — stability + null calibration + engine agreement

No labels. No regimes. Just geometry.
```

---

## PR Update: Enforce Glossary

Add to **PR 1 (HSB Foundation)**:
```
## Add PRISM_GLOSSARY.md to repo root

Include the full glossary as a markdown file.

## Update prism/core/semantic_guard.py

Add glossary-aligned checks:

FORBIDDEN_TOKENS = [
    "regime", "regimes",
    "bull", "bear",
    "crisis", "risk-on", "risk-off",
    "stressed", "calm",
    "normal state", "stress state"
]

# Correct terminology (for documentation, not enforcement)
CANONICAL_TERMS = {
    "distance": "displacement",
    "stress": "displacement",
    "risk": "excursion" or "escalation_hazard",
    "recovery": "reversion",
    "crisis": "deformation_acceleration",
    "regime_change": "band_crossing",
}
```

---

## One Signature Term Proposal

GPT suggested coining a "Mahalanobis moment" — one original PRISM term that becomes associated with the framework.

**Proposal: "Excursion Depth-Duration" (EDD)**

A single metric combining how far and how long:
```
EDD = ∫ max(D(t) - r, 0) dt over excursion