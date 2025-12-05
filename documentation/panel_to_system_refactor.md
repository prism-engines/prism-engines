# Refactor Request: Rename `panel` → `system` Across SQL + Python

This document is for Claude Code / Copilot to implement as a focused refactor on the `sql` branch of **rudder-research/prism-engine**.

The core change:  
> The `panel` column in the **indicators** table currently represents the *system/domain* (e.g. `finance`, `climate`, `chemistry`, `anthropology`).  
> We want to rename this concept everywhere to `system` for clarity and future multi-domain support.

Please implement the changes below and open a PR against `main` (from the `sql` branch or a new feature branch off `sql`).

---

## 1. SQL Schema: Rename `panel` → `system`

**Files to focus on:**
- `data/sql/prism_schema.sql`
- Any other SQL schema or helper scripts under `data/sql/`

### 1.1. Update schema definition

In the **`indicators`** table definition, rename the `panel` column to `system`:

- Change the column name from `panel` to `system` everywhere in the schema.
- Keep the meaning the same: `system` = top‑level domain for the indicator (`finance`, `climate`, etc.).

Example (pseudocode, adapt to existing schema):

```sql
CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    system TEXT NOT NULL,
    frequency TEXT NOT NULL,
    source TEXT,
    units TEXT,
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 1.2. Migration script for existing DBs

Add a **migration SQL script** under a migrations folder, for example:

- `data/sql/migrations/001_rename_panel_to_system.sql`

The migration should:

1. **Rename the column** from `panel` → `system` in the `indicators` table.
2. Handle SQLite correctly (if `ALTER TABLE ... RENAME COLUMN` is not available or we want to be safe, do the “create new table / copy / rename” pattern).
3. Preserve all existing data.

Rough idea (you can choose the best SQLite‑compatible approach):

```sql
-- If supported:
ALTER TABLE indicators RENAME COLUMN panel TO system;
```

Or the safer 3‑step rewrite version if needed.

### 1.3. Foreign keys and indexes

Please verify if any other tables reference `indicators.panel` indirectly or if there are indexes on `panel`:

- If there is an index on `panel`, recreate it on `system`, for example:

```sql
CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system);
```

- If there are foreign keys referencing `indicators.id`, that’s fine; just make sure the migration does not break any constraints.

---

## 2. Python Refactor: Parameters & Queries

**Main file:**
- `data/sql/prism_db.py`

### 2.1. Rename function parameters

Everywhere in `prism_db.py` where a function takes a `panel` argument, change it to `system`:

- `add_indicator(name, panel, frequency, ...)` → `add_indicator(name, system, frequency, ...)`
- `write_dataframe(..., panel=..., ...)` → `write_dataframe(..., system=..., ...)`
- `load_indicator(..., panel=..., ...)` (if present) → use `system` instead.

Change internal variable names accordingly (`panel` → `system`).

### 2.2. Update SQL inside Python

All SQL statements embedded in Python that reference the column `panel` should now reference `system` instead.

Examples (adapt to current code):

```python
INSERT INTO indicators (name, system, frequency, source, units, description)
VALUES (?, ?, ?, ?, ?, ?)
```

```python
SELECT id, name, system, frequency, source, units, description
FROM indicators
WHERE name = ?
```

### 2.3. Deprecation guard for old `panel` argument

For backwards compatibility, allow callers to still pass a `panel` keyword *for now*, but emit a warning and map it to `system`.

Example pattern:

```python
import warnings

SYSTEM_TYPES = ["finance", "climate", "chemistry", "anthropology", "biology", "physics"]

def add_indicator(
    name: str,
    system: str | None = None,
    *,
    panel: str | None = None,
    frequency: str = "daily",
    source: str | None = None,
    units: str | None = None,
    description: str | None = None,
):
    if panel is not None:
        warnings.warn(
            "'panel' is deprecated; use 'system' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # If system not explicitly provided, adopt panel value
        if system is None:
            system = panel

    if system is None:
        raise ValueError("`system` is required (e.g. 'finance', 'climate').")

    if system not in SYSTEM_TYPES:
        raise ValueError(f"Unknown system: {system}. Valid values: {SYSTEM_TYPES}")

    # existing insert logic here, using `system`
```

Follow the same pattern for other key functions if they previously accepted `panel`.

> **Goal:** existing code that still uses `panel=` does not immediately break, but we nudge toward `system=` going forward.

### 2.4. Define `SYSTEM_TYPES` constant

Near the top of `prism_db.py`, define a shared constant listing allowed system names, e.g.:

```python
SYSTEM_TYPES = [
    "finance",
    "climate",
    "chemistry",
    "anthropology",
    "biology",
    "physics",
]
```

Feel free to put this in a config module if that fits better, but the important part is centralizing the allowed values.

---

## 3. Repo‑Wide Search & Replace

Please run a repository search for the string `"panel"` to catch all call sites and configs that refer to this domain concept.

Example search command (for reference):

```bash
grep -r "panel" .   --include="*.py"   --include="*.ipynb"   --include="*.yaml"   --include="*.yml"   --include="*.json"   --include="*.md"
```

Update usages where **`panel` clearly means “system/domain”** and should now be named `system`. Be careful not to touch unrelated uses of the word “panel” (e.g., UI panels, notebooks, or comments that refer to data panels generically).

---

## 4. Documentation Updates

Update any documentation that describes this concept:

- `data/sql/README_SQL.md`
- Any other README or design docs that describe a “panel” as the domain level (e.g. finance vs climate).

Replace that terminology with **`system`**, and add a brief explanation such as:

> **system**: top‑level domain for an indicator (e.g. `finance`, `climate`, `chemistry`). A single database can store multiple systems side by side.

If relevant, note that older code may still reference `panel`, but that it is deprecated in favor of `system`.

---

## 5. Testing & Validation

Please add or update lightweight tests / smoke checks to confirm the refactor works.

At minimum:

1. Run the migration against a test copy of the SQLite DB.
   - For Jason’s environment, the live DB path is provided by the `PRISM_DB` environment variable. Locally you can create a temporary DB for tests.
2. Use Python to:
   - Create an indicator with `system="finance"` (or `"equity"` if that’s the sub‑system label used).
   - Write a small DataFrame with values for that indicator.
   - Load it back via `load_indicator` and confirm the data round‑trips correctly.
3. Confirm that calling `add_indicator(..., panel="finance")` still works but raises a `DeprecationWarning`.

Example quick check (pseudo‑test):

```python
from data.sql.prism_db import add_indicator, write_dataframe, load_indicator

import pandas as pd

df = pd.DataFrame(
    {
        "date": ["2024-01-01", "2024-01-02"],
        "value": [100.0, 101.5],
    }
)

add_indicator(
    name="TEST_TICKER",
    system="finance",
    frequency="daily",
    source="dummy",
    units="index",
)

write_dataframe(
    df,
    indicator_name="TEST_TICKER",
    system="finance",
    frequency="daily",
    source="dummy",
)

roundtrip = load_indicator("TEST_TICKER")
print(roundtrip.head())
```

---

## 6. Git & PR Details

- Base branch: `main`
- Working branch: you can either keep using `sql` or create a new branch (e.g. `refactor/rename-panel-to-system`) off `sql`.
- Commit message suggestion:
  - `Refactor: rename panel → system in SQL + Python`
- PR title suggestion:
  - **“Refactor: Rename panel → system (multi‑system support)”**

In the PR description, please summarize:

- Schema change (`panel` → `system`).
- Migration script added.
- Python refactor (functions & queries).
- Deprecation handling for `panel` argument.
- New `SYSTEM_TYPES` constant.
- Any docs/tests added or updated.

That’s it — the goal is a clean, consistent move from `panel` to `system` across schema, helper library, and call sites, with a gentle deprecation path so existing code doesn’t immediately break.
