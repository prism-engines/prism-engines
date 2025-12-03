# PR #2 – Prism Registry System (system, market, economic)

## 1. Purpose
Implement the **registry-driven architecture** that acts as the “quarterback” for Prism. All paths, datasets, and series-level metadata must be centralized in JSON registries.

## 2. Scope
- Implement:
  - `system_registry.json`
  - `market_registry.json`
  - `economic_registry.json`
- Add loader and validator utilities.
- Do not yet implement full fetch logic; this PR focuses on **data description**, not ingestion.

## 3. Files to Create / Modify

### 3.1 data_fetch/system_registry.json
Example minimal structure:

```json
{
  "paths": {
    "data_raw": "data/raw",
    "data_clean": "data/clean",
    "db_path": "db/prism.db",
    "logs_dir": "logs"
  },
  "registries": {
    "market": "data_fetch/market_registry.json",
    "economic": "data_fetch/economic_registry.json"
  },
  "panel": {
    "master_panel_path": "data/clean/panel_master.csv"
  }
}
```

### 3.2 data_fetch/market_registry.json
Example structure:

```json
{
  "SPY": {
    "fetch_type": "yahoo",
    "enabled": true,
    "frequency": "daily",
    "tables": {
      "prices": "market_prices",
      "dividends": "market_dividends",
      "tri": "market_tri"
    },
    "use_column": "tri_value",
    "required_fields": ["price", "dividend"],
    "notes": "Core S&P 500 ETF"
  }
}
```

### 3.3 data_fetch/economic_registry.json
Example structure:

```json
{
  "CPIAUCSL": {
    "fetch_type": "fred",
    "enabled": true,
    "frequency": "monthly",
    "table": "econ_values",
    "code": "CPIAUCSL",
    "use_column": "value_z",
    "revision_tracking": true,
    "transformations": ["yoy", "zscore"],
    "notes": "Headline CPI, seasonally adjusted"
  }
}
```

### 3.4 utils/db_connector.py
- Add helper functions to:
  - Open a SQLite connection using the path from `system_registry.json`.
  - Possibly provide a context manager wrapper.

### 3.5 utils/fetch_validator.py
- Add functions to validate registry entries:
  - Check required keys exist (e.g., `fetch_type`, `frequency`, `table`, `use_column`).
  - Check column names and table names are sane.

## 4. Behavior
- No external calls yet.
- Registry loaders should:
  - Read JSON files.
  - Validate schema.
  - Raise clear exceptions if invalid.

## 5. Out of Scope
- Actual fetching of market/economic data.
- Database migrations.
- Panel assembly.

## 6. Acceptance Criteria
- Registries can be loaded and validated without errors.
- All paths and table names are centralized in the registries.
- No hardcoded table or path references remain in the code; they must be routed through the registries.
