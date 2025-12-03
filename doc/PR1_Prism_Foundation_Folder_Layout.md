# PR #1 – Prism Engine Foundation & Folder Layout

## 1. Purpose
Establish the **foundational folder structure** for Prism Engine and remove any remaining VCF-era layout assumptions, without changing functional logic yet.

## 2. Scope
- Create the new top-level directory layout.
- Move or create empty placeholder files where needed.
- Do **not** implement business logic beyond what is necessary to make imports and paths valid.

## 3. Target Folder Structure
Create or align the repo to:

```text
prism-engine/
│
├── data_fetch/
│   ├── fetch_market_data.py          # placeholder (minimal no-op or stub)
│   ├── fetch_economic_data.py        # placeholder (minimal no-op or stub)
│   ├── market_registry.json          # placeholder JSON {} or minimal structure
│   ├── economic_registry.json        # placeholder JSON {} or minimal structure
│   └── system_registry.json          # placeholder JSON {} or minimal structure
│
├── db/
│   ├── prism.db                      # may be created later; ensure path assumed
│   └── migrations/                   # empty for now or with stub .md
│
├── data/
│   ├── raw/                          # storage for unprocessed data
│   └── clean/                        # storage for normalized data / panels
│       └── .gitkeep
│
├── panel/
│   ├── build_panel.py                # placeholder
│   ├── transforms_market.py          # placeholder
│   ├── transforms_econ.py            # placeholder
│   └── validators.py                 # placeholder
│
├── engine/
│   ├── prism_macro_engine.py         # existing or stub
│   ├── prism_market_engine.py        # existing or stub
│   ├── prism_stress_engine.py        # existing or stub
│   ├── prism_ml_engine.py            # existing or stub
│   └── __init__.py
│
├── utils/
│   ├── date_cleaner.py               # placeholder
│   ├── number_cleaner.py             # placeholder
│   ├── fetch_validator.py            # placeholder
│   ├── db_connector.py               # placeholder
│   └── log_manager.py                # placeholder
│
├── logs/
│   ├── .gitkeep
│
└── Start/
    ├── run_fetch_all.py              # placeholder
    ├── run_panel_build.py            # placeholder
    └── run_prism_engine.py           # placeholder
```

## 4. Requirements
- No functional behavior change beyond what is required to keep imports and basic scripts runnable.
- All new Python files can initially contain simple stubs such as:
  ```python
  """Stub file created as part of Prism foundational layout."""
  ```
- All paths must be relative to the repo root (`prism-engine/`) and **not hardcoded to system-specific paths**.

## 5. Out of Scope
- Real fetch logic.
- Real registry population.
- Real DB migrations.
- Real panel-building or engine enhancements.

## 6. Acceptance Criteria
- Repo matches the folder layout above.
- Existing code (if any) imports successfully under the new structure.
- No references remain to legacy VCF folders or names in the layout.
