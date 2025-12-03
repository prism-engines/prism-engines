# PR #6 – Prism Engine Integration with Panel & Registries

## 1. Purpose
Connect the existing Prism engines to the new panel and registry-driven architecture so that engines no longer hardcode paths, filenames, or columns.

## 2. Scope
- Update:
  - `engine/prism_macro_engine.py`
  - `engine/prism_market_engine.py`
  - `engine/prism_stress_engine.py`
  - `engine/prism_ml_engine.py` (if applicable)
- Ensure all engines:
  - load panel data from the path specified in `system_registry.json`
  - use registry metadata (e.g., to know what each column represents, or to map series IDs to Pillars/Lenses later)

## 3. Integration Details

### 3.1 Panel Loading
- Implement a small helper (either inside each engine or shared in `utils/`) to:
  - Read `system_registry.json` to locate `panel_master_path`.
  - Load the panel as a DataFrame (CSV or future Parquet).

### 3.2 Column Interpretation
- For now, engines can treat each column as a numeric series keyed by:
  - ticker for market series
  - code for economic series

- Future work (not in this PR) can:
  - Map columns to Pillars, Lenses, or Factor groups via additional registry files.

### 3.3 Backward Compatibility
- If any existing engine code assumed:
  - direct CSV paths,
  - old filenames, or
  - legacy VCF structures

then:
- Replace these assumptions with:
  - registry-driven lookups, or
  - new helper functions.

## 4. Out of Scope
- Implementing new engine types or algorithms.
- Introducing new Pillar/Lens definitions.
- Any major refactor of the numerical logic inside engines.

## 5. Acceptance Criteria
- All engines can be run using the new panel path.
- No engine references hardcoded old paths or filenames.
- Engines remain numerically consistent with prior behavior where applicable (modulo improved data quality).
