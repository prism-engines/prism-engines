PRISM CANONICAL LIMITS
=====================

Purpose
-------
Define the hard analytical limits of the PRISM system to ensure:
- bounded computation
- interpretable geometry
- avoidance of infinite derivative chains
- practical execution without supercomputers

These limits are not philosophical. They are enforced by:
- computational complexity
- identifiability
- stability of geometry


CORE AXIOM
----------
PRISM only computes geometry:
1) Within a single system/domain
2) Of indicators relative to that system
3) Of the system relative to itself over time

Cross-system analysis is always selective.


CANONICAL SCOPE 1: INDICATOR GEOMETRY (WITHIN SYSTEM)
-----------------------------------------------------
Definition:
- A system is a closed set of indicators operating under a shared domain.

Canonical computation:
(time_window, indicator, lens) -> metric values

From this, derive:
- Indicator feature vectors (lens outputs concatenated)
- Local indicator geometry within a time window

Permitted operations:
- Indicator distance / similarity
- Neighborhoods (k-NN)
- Clustering
- Stability and drift over time

Forbidden operations:
- Global all-pairs indicator geometry without bounds
- Unwindowed time aggregation
- Indicator x Indicator x Time full tensors


CANONICAL SCOPE 2: SYSTEM GEOMETRY (SELF-RELATIVE)
-------------------------------------------------
Definition:
- The system itself is treated as a moving point in its own state space.

Canonical computation:
(time_window) -> system state vector

System state may include:
- aggregate lens statistics
- lens agreement metrics
- compression / rank estimates
- stability / volatility measures

Permitted outputs:
- system trajectory over time
- regime transitions
- deformation / stress scores

The system is never embedded in a universal space.
It only exists relative to its own historical states.


CANONICAL CROSS-SYSTEM COMPARISON (SELECTIVE)
---------------------------------------------
Rule:
Cross-system geometry is ONLY computed after selection.

Procedure:
1) Compute influence scores for indicators within each system.
2) Select top-K indicators per system.
3) Compare only these selected indicators across systems.

K is small by design (e.g. 10-50).

Rationale:
- avoids combinatorial explosion
- preserves interpretability
- reflects practical scientific workflows

Forbidden:
- Full cross-product of indicators across systems
- Global multi-system tensors


STOP CONDITIONS (HARD LIMITS)
-----------------------------
Further derivative layers are not computed when:

1) Geometry becomes unstable
   - nearest neighbors change drastically across small window shifts

2) Distances collapse
   - pairwise distances converge to similar values

3) Structure is not compressible
   - no dominant modes / low-rank structure

4) Multiple configurations yield equivalent geometry
   - loss of identifiability

When these occur, geometry is considered shapeless.


DESIGN PRINCIPLE
----------------
PRISM does not attempt to compute everything.
PRISM computes what can be stably inferred.

Selection precedes expansion.
