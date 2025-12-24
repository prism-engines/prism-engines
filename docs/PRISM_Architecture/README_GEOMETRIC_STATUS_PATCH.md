PRISM Geometric Status Patch (drop-in)

What this zip contains:
- PRISM_Geometric_State_And_Meaning.md (root-level design note)
- docs/GEOMETRIC_STATUS_ACTION_ITEM.md (action item + wiring plan)
- prism/geometry/geom_status.py (library)
- prism/geometry/geom_status_cli.py (CLI)
- examples/geom_metrics_example.csv (example input)

How to use (from your repo root):
1) Unzip into repo root (it will merge docs/ and prism/geometry/).
2) In your venv:
   pip install -e .
3) Run the example:
   python -m prism.geometry.geom_status_cli --input examples/geom_metrics_example.csv --out output/geom_status_timeline.csv --json output/geom_status_current.json

Next step in-engine:
- produce output/geom_metrics.csv from your lens artifacts
- feed it into the CLI to get a timeline + current state
