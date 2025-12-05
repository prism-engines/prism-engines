# Prism Engine – Data Pipeline, Registry, and DB Overhaul (PR Specification)

## 1. Purpose
This PR establishes the foundational architecture for Prism’s data ingestion, normalization, storage, and panel-building system. It replaces all legacy VCF structures and introduces a unified, expandable, registry‑driven pipeline.

## 2. High-Level Goals
- Separate **market** and **economic** pipelines  
- Introduce a **system registry** as the single source of truth  
- Add **multi-column support** for each data series (raw, normalized, transformed, engine‑specific)  
- Implement strict **fetch normalization layers**  
- Standardize the **database schema** for long-term scalability  
- Consolidate panel building under registry control  
- Clean, deterministic **folder structure**

## 3. New Folder Structure
```
prism-engine/
│
├── data_fetch/
│   ├── fetch_market_data.py
│   ├── fetch_economic_data.py
│   ├── market_registry.json
│   ├── economic_registry.json
│   └── system_registry.json
│
├── db/
│   ├── prism.db
│   └── migrations/
│       ├── 001_split_market_econ.sql
│       ├── 002_create_market_tables.sql
│       ├── 003_create_economic_tables.sql
│       ├── 004_add_multi_column_support.sql
│       ├── 005_enforce_date_normalization.sql
│       └── 006_add_metadata_tables.sql
│
├── data/
│   ├── raw/
│   └── clean/
│       └── panel_master.csv
│
├── panel/
│   ├── build_panel.py
│   ├── transforms_market.py
│   ├── transforms_econ.py
│   └── validators.py
│
├── engine/
│   ├── prism_macro_engine.py
│   ├── prism_market_engine.py
│   ├── prism_stress_engine.py
│   ├── prism_ml_engine.py
│   └── …
│
├── utils/
│   ├── date_cleaner.py
│   ├── number_cleaner.py
│   ├── fetch_validator.py
│   ├── db_connector.py
│   └── log_manager.py
│
├── logs/
│   ├── fetch_log.txt
│   ├── panel_build_log.txt
│   └── error_log.txt
│
└── Start/
    ├── run_fetch_all.py
    ├── run_panel_build.py
    └── run_prism_engine.py
```

## 4. Registry System (Quarterback Architecture)

### 4.1 system_registry.json
Controls:
- paths  
- db locations  
- registry file locations  
- logging settings  
- output locations  

### 4.2 market_registry.json
Defines:
- tickers  
- fetch type  
- required fields  
- table mappings  
- which column engines should use (ex: TRI vs raw price)

### 4.3 economic_registry.json
Defines:
- economic series  
- source (FRED/BLS/IMF/etc)  
- revision rules  
- required transformations  

## 5. Multi-Column Data Support
Each series now supports:
- raw  
- normalized  
- yoy/mom/returns  
- z‑scores  
- log transforms  
- engine‑specific derived values  

Engines select their preferred column via:
```
"use_column": "value_z"
```

## 6. Fetch Normalization Requirements
All fetch scripts must:
1. Normalize dates → YYYY-MM-DD  
2. Normalize column names  
3. Remove footer garbage  
4. Remove duplicate dates  
5. Convert all numerics  
6. Repair 2‑digit year formats  
7. Drop invalid rows  
8. Validate shape before DB insert  
9. Execute registry‑driven validation rules  

## 7. Database Schema (New)
### market tables:
- market_prices  
- market_dividends  
- market_tri  
- market_meta  

### economic tables:
- econ_series  
- econ_values (w/ revision_asof)  
- econ_meta  

### multi-column migration:
Adds:
- value_raw  
- value_z  
- value_yoy  
- value_mom  
- log_value  
- tri  
- pct_change  

## 8. Panel Builder
The panel builder:
- reads system registry  
- loads market registry  
- loads economic registry  
- determines which column each series uses  
- assembles unified panel  
- writes to `data/clean/panel_master.csv`

No hardcoded paths. All registry-driven.

## 9. Testing Requirements
Claude Code must produce:
- fetch tests  
- DB schema validation tests  
- panel assembly tests  
- frequency compliance tests  
- duplicate handling tests  
- date normalization tests  

## 10. Migration Plan
1. Create new tables  
2. Migrate existing market data  
3. Run fetchers to repopulate new data  
4. Build first unified panel  
5. Verify shape and alignment  
6. Remove legacy VCF files (if applicable)

## 11. PR Summary
This PR introduces a complete, registry‑driven redesign of Prism’s data architecture, ensuring long-term stability, scalability, and correctness.

All engines remain functional and will now reference a unified, validated, multi-column database backed by strict fetch normalization.

