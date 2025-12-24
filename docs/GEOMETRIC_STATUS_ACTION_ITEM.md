# Action Item: Geometric Status Timeline and Current State

Goal:
Build a thin, explicit "meaning layer" that translates geometric change into a timeline and a current-state readout.

Core outputs:
1) Timeline CSV (per date/window):
   - curvature
   - coherence
   - localization
   - complexity
   - d_curvature, d_coherence, d_localization (deltas)
   - state_label (healthy_complex, pathological_complex, stable_coherent, transition)

2) Current-state summary (printed to console and optionally JSON):
   - latest values
   - recent deltas (e.g., last N windows)
   - top warning flags

Inputs (initial version):
- A CSV you export from the engine with these columns:
  date, curvature, coherence, localization, complexity
  - date in YYYY-MM-DD (or timestamp)
  - each metric in a consistent 0..1 scale (or standardized)

Later (v2+):
- derive these metrics directly from lens artifacts:
  - coherence from multi-lens agreement / correlation stability
  - curvature from deformation acceleration measures
  - localization from contribution concentration (HHI/Gini over indicator "mass")
  - complexity from interaction-geometry instability (coherence volatility + phase decoherence)

Wiring plan:
- prism/geometry/geom_status.py (library)
- prism/geometry/geom_status_cli.py (CLI)

CLI example:
python -m prism.geometry.geom_status_cli --input output/geom_metrics.csv --out output/geom_status_timeline.csv --json output/geom_status_current.json

Notes:
- This is intentionally "thin" and explicit so it can evolve as the geometry engine matures.
