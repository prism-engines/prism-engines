# PR #5 – Prism Panel Builder & Column Selection

## 1. Purpose
Implement the **panel building layer** that reads from the database and registries to produce a unified, engine-ready panel (e.g., `panel_master.csv`), with per-series control over which column is used.

## 2. Scope
- Implement panel-building logic in `panel/build_panel.py`.
- Implement transformation helpers in:
  - `panel/transforms_market.py`
  - `panel/transforms_econ.py`
- Implement validation helpers in `panel/validators.py`.

## 3. Behavior Overview

### 3.1 Registry-Driven Column Selection
- For each market series in `market_registry.json`:
  - Read `tables` mapping and `use_column` (e.g., `tri_value`, `price`, `ret`).
- For each economic series in `economic_registry.json`:
  - Read `table` and `use_column` (e.g., `value_z`, `value_yoy`, `value_raw`).

### 3.2 Panel Assembly
- Query DB using `db_connector` for each series.
- Convert each selected series into a standardized time-series:
  - index: `date`
  - column name: some consistent identifier (e.g., ticker or code)
- Outer-join or left-join all series on `date`.
- Optionally apply:
  - Alignment logic (fill with NaN where missing).
  - Basic sanity checks (e.g., non-empty rows).

- Write final panel to:
  - Path from `system_registry["panel"]["master_panel_path"]`.

## 4. Files to Create / Modify

### 4.1 panel/build_panel.py
Implement functions:
- `build_panel()`:
  - Loads registries.
  - Queries the DB.
  - Constructs a combined DataFrame.
  - Calls validators.
  - Writes output.

### 4.2 panel/transforms_market.py
- Optional helper functions for:
  - Aligning different market series.
  - Additional smoothing/derived columns if needed later.

### 4.3 panel/transforms_econ.py
- Optional helper functions for:
  - Aligning economic series.
  - Handling different frequencies (e.g., forward-fill monthly onto daily calendar if desired).

### 4.4 panel/validators.py
Implement:
- `validate_panel(df)`:
  - Check date index is sorted.
  - Check no duplicate dates.
  - Check at least N rows.
  - Log any columns with too many NaNs.

## 5. Out of Scope
- Engine internals.
- Any ML or advanced transforms beyond what’s required for basic panel creation.

## 6. Acceptance Criteria
- `build_panel.py` runs successfully end-to-end on existing DB content.
- `panel_master.csv` is created with expected shape and column names.
- Logs show successful assembly and any warnings for sparse series.
