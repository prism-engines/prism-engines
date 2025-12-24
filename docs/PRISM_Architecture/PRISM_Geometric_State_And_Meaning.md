# PRISM: Geometric State and Meaning

## 1. From indicators to fields
In PRISM, indicators are not "just numbers." They behave like vectors in a shared state space:
- magnitude (how strong)
- direction (what kind of force / sign relative to meaning)
- domain (what axis / subsystem it acts on)
- phase (timing/lag relative to others)

A naive sum can destroy information (e.g., +1 and -1 producing 0). PRISM avoids scalar cancellation by treating indicator interaction as a field.

## 2. Deformation as system state
Deformation is not an arithmetic sum of indicator values. It is the resultant geometric change of the system under interacting vector forces.

Opposition does not "cancel." It can increase tension/strain. Alignment can increase momentum/coherence. Orthogonality can increase texture/complexity. The system state is the geometry of deformation.

## 3. Dynamic complexity
Complexity is dynamic: it emerges from changing relationships (coupling, phase, coherence), not from the count of indicators or raw volatility.

A system can be:
- high volatility, low complexity (coherent panic)
- low volatility, high complexity (latent fracture)

PRISM should track complexification (how interaction geometry changes), not only a static complexity score.

## 4. Healthy vs pathological complexity
Healthy dynamic complexity:
- relationships change but remain structured
- stress/opposition exists but is distributed
- lenses may disagree, but in patterned ways
- deformation is elastic (recovers)

Pathological complexity:
- relationships change erratically
- coupling weakens or snaps
- stress localizes into fewer nodes/domains
- deformation becomes jagged/plastic (does not recover)

A practical discriminator:
Healthy complexity preserves global coherence under local change.
Pathological complexity loses global coherence under local stress.

## 5. Geometric change primitives (X, Y, Z)
As geometry changes over time, PRISM should detect and present a small set of field-level primitives:

X: Curvature change
- what: acceleration/deceleration of deformation
- meaning: bending faster (pressure building), or relaxing

Y: Coherence shift
- what: lenses/relationships moving into or out of agreement
- meaning: coordination vs fragmentation (healthy vs pathological complexity)

Z: Stress localization
- what: deformation concentrating into fewer indicators/domains
- meaning: fragility / single-point failure risk

## 6. Detection vs interpretation
Detection answers: what changed geometrically?
Interpretation answers: what does it mean?

PRISM should generate human-readable state descriptors from X/Y/Z (and supporting metrics) rather than forcing humans to interpret raw scalar outputs.

## 7. Presentation philosophy
Presentation is not "pretty charts." It is meaning.
PRISM should surface timeline + current state in language like:
- "Curvature accelerating with declining coherence."
- "Opposition rising but still distributed (elastic)."
- "Stress localizing in macro liquidity domain."

## 8. Action item
Implement a geometric status timeline:
- ingest per-time-window lens outputs (or derived metrics)
- compute X (curvature change), Y (coherence shift), Z (stress localization)
- classify state (healthy vs pathological complexity)
- output a time series + a current-state summary
