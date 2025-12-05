# PR: Add PRISM SQL Database Layer (Option A – Claude SQL Plan)

## Summary

This PR adds a **first-class SQL storage layer** for the PRISM engine using a dedicated SQLite database.  
It follows the **Option A / Claude SQL plan**, which not only stores raw indicators but also supports engine outputs (lenses, windows, regimes, coherence events, etc.).

- Introduces a **unified schema** for indicators, values, lenses, temporal windows, rankings, and coherence events.
- Adds a **Python helper library** (`prism_db.py`) to read/write data from Python engines.
- Keeps the actual SQLite database file **outside the GitHub repo** (on Google Drive), so code is versioned but data is not.
- Provides a clear path for future engines (finance, climate, chemistry, anthropology, etc.) to write and query panel data consistently.

---

## Files Added

These files should be added to the repo (recommended location: `utils/sql/` or `04_data_clean/sql/` – adjust as needed to match your structure):

- `prism_schema.sql`  
  - Defines all tables, indexes, and views for:
    - `indicators`
    - `indicator_values`
    - `lenses`
    - `windows`
    - `lens_results`
    - `consensus`
    - `regime_stability`
    - `coherence_events`
    - `engine_runs`
  - Includes unique constraints and indexes to avoid duplicates and speed up queries.

- `prism_db.py`  
  - Python helper module for all interaction with the database:
    - `init_database()` – create tables from `prism_schema.sql` if they don’t exist.
    - `add_indicator(...)` – register a new indicator with panel and frequency.
    - `write_values(...)` – write time-series rows for a given indicator.
    - `write_dataframe(...)` – one-shot helper to take a pandas DataFrame with `date` and `value` and store it.
    - `load_indicator(...)` – load a single indicator as a pandas DataFrame.
    - `load_multiple(...)` – load several indicators into a pivoted DataFrame (columns = indicators).
    - `query(sql)` – run arbitrary SQL and return a DataFrame.

- `README_SQL.md`  
  - Short, task-focused documentation for:
    - Creating the database directory.
    - Setting the `PRISM_DB` environment variable.
    - Initializing the database.
    - Example usage for import, storage, and queries.

(If desired, you can also include `example_usage.py` as a living demo script under something like `Start/Examples/`.)

---

## Database Location and Environment Variable

**The database file itself is _not_ part of the repo**.  
It lives in your Google Drive folder so it is:
- Backed up
- Shared across Chromebook / Mac / Windows
- Protected from accidental deletion via Git operations

**Confirmed location for Jason:**

```bash
/mnt/chromeos/GoogleDrive/MyDrive/prismsql/prism.db
```

Set the environment variable in your shell config (`~/.bashrc`, `~/.zshrc`, or equivalent):

```bash
export PRISM_DB="/mnt/chromeos/GoogleDrive/MyDrive/prismsql/prism.db"
```

Then reload:

```bash
source ~/.bashrc   # or ~/.zshrc
```

If `PRISM_DB` is not set, `prism_db.py` can fall back to a default like:

```text
~/prismsql/prism.db
```

(Adjust fallback logic in `prism_db.py` if you prefer a different default.)

---

## Schema Overview (Conceptual)

### 1. Input Layer (Indicators & Values)

**Table: `indicators`**  
One row per indicator (per panel, per frequency).

- `name` – e.g. `"SPY"`, `"CPI"`, `"DXY"`
- `panel` – e.g. `"equity"`, `"macro"`, `"climate"`, `"chemistry"`, `"anthropology"`
- `frequency` – e.g. `"daily"`, `"monthly"`, `"quarterly"`, `"ms"`, `"century"`
- `source` – `"Yahoo"`, `"FRED"`, `"custom"`
- `units` – optional description (e.g. `"index"`, `"percent"`, `"temperature"`)

**Table: `indicator_values`**  
Time-series values linked to `indicators` by `indicator_id`:
- `date` – stored as text (ISO string)
- `value` – primary numeric value
- optional extra numeric fields for transformed values

This layer is **panel-agnostic**: finance, climate, chemistry, anthropology, etc. all use the same tables.

---

### 2. Engine Output Layer

The schema adds explicit support for PRISM/VCF engines and temporal structure:

- **`lenses`** – defines your lenses (e.g. `"trend"`, `"volatility"`, `"coherence"`, `"macro_stress"`).
- **`windows`** – time windows used for temporal analysis (e.g. `2005-01-01 → 2010-12-31`).
- **`lens_results`** – per-lens, per-window scores/rankings for each indicator.
- **`consensus`** – aggregated scores when combining multiple lenses.
- **`regime_stability`** – stability metrics over time.
- **`coherence_events`** – moments where lenses agree / regimes shift.
- **`engine_runs`** – metadata for reproducibility (run id, timestamp, parameters, code version).

This allows you to:

- Track evolution of rankings over time.
- Identify regime transitions.
- Log coherence events.
- Reproduce any engine run later with the same DB.

---

## Usage Examples (Python)

### Initialize Database

```python
from prism_db import init_database
init_database()  # reads prism_schema.sql and creates tables if needed
```

### Register an Indicator and Write Values

```python
import pandas as pd
from prism_db import add_indicator, write_values, write_dataframe

# Example: SPY daily prices
df = pd.DataFrame({
    "date": [...],
    "value": [...],
})

# Option 1: register then write
add_indicator("SPY", panel="equity", frequency="daily", source="Yahoo")
write_values("SPY", df)

# Option 2: one-shot helper
write_dataframe(df, "SPY", panel="equity", source="Yahoo", frequency="daily")
```

### Read Back Data

```python
from prism_db import load_indicator, load_multiple

# Single indicator
spy = load_indicator("SPY", start_date="2010-01-01")

# Multiple indicators, pivoted
panel_df = load_multiple(["SPY", "DXY", "AGG", "TLT"])
```

### Run Arbitrary Queries

```python
from prism_db import query

# All indicators
indicators_df = query("SELECT * FROM indicators")

# Count data points per indicator
counts = query(
    "SELECT i.name, COUNT(*) AS n_points "
    "FROM indicator_values iv "
    "JOIN indicators i ON iv.indicator_id = i.id "
    "GROUP BY i.id "
    "ORDER BY n_points DESC;"
)
```

Engine outputs (lens scores, regimes, coherence events) are written/read using similar helpers you can add to `prism_db.py` as needed.

---

## How This Fits the PRISM Phases

- **Phase II (Bulletproofing & Organization)**  
  - All engines write to a single, coherent storage layer.
  - No more scattered CSVs; everything is queryable and reproducible.

- **Phase III (Visualization & Reporting)**  
  - Dashboards can query from `lens_results`, `consensus`, `regime_stability`, and `coherence_events` directly.
  - Any visualization layer (Streamlit, Dash, Power BI, etc.) can sit on top of the SQLite database.

- **Phase IV (Applications)**  
  - This schema becomes the core “data contract” for apps and services built on top of PRISM/VCF.

---

## Notes for Reviewers (Jason + Claude + others)

- The `.db` file itself is **not versioned**; only schema & helpers are.
- This PR is **purely additive** – it does not yet rip out any existing CSV-based flows.
- Engines can gradually migrate to using `prism_db.py` as the canonical I/O layer.
- Future PRs can:
  - Add more helper functions for engine outputs.
  - Introduce migrations or upgrades to the schema.
  - Integrate this SQL layer into existing orchestration pipelines.

---

## Post-Merge Checklist for Jason

1. Create the DB directory on ChromeOS (if not already):
   ```bash
   mkdir -p /mnt/chromeos/GoogleDrive/MyDrive/prismsql
   ```

2. Set your environment variable:
   ```bash
   export PRISM_DB="/mnt/chromeos/GoogleDrive/MyDrive/prismsql/prism.db"
   ```

3. Copy `prism_schema.sql` and `prism_db.py` into the repo at the agreed location (e.g. `utils/sql/`).

4. Initialize the database from Python:
   ```python
   from prism_db import init_database
   init_database()
   ```

5. Start migrating one or two indicators from CSV → SQL to validate the workflow.

