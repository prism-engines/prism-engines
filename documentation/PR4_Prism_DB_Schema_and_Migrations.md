# PR #4 – Prism Database Schema & Migrations (Market + Economic + Multi-Column)

## 1. Purpose
Define and implement the **SQLite database schema** used by Prism for all market and economic data, including support for multi-column series (raw, transformed, and engine-specific values).

## 2. Scope
- Create core tables for:
  - Market data
  - Economic data
  - Metadata
- Implement migrations under `db/migrations/`.
- Integrate with `db_connector` and registry paths.
- No panel logic yet.

## 3. Tables to Implement

### 3.1 market_prices
Columns:
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `ticker` (TEXT, indexed)
- `date` (TEXT, ISO `YYYY-MM-DD`, indexed)
- `price` (REAL)
- `ret` (REAL, optional daily return)
- `price_z` (REAL, optional normalized field)
- `price_log` (REAL, optional log(price))

Unique constraint:
- (`ticker`, `date`)

### 3.2 market_dividends
Columns:
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `ticker` (TEXT, indexed)
- `date` (TEXT, ISO `YYYY-MM-DD`, indexed)
- `dividend` (REAL)

Unique constraint:
- (`ticker`, `date`)

### 3.3 market_tri
Columns:
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `ticker` (TEXT, indexed)
- `date` (TEXT, ISO `YYYY-MM-DD`, indexed)
- `tri_value` (REAL)
- optional derived fields: `tri_z`, `tri_log`

Unique constraint:
- (`ticker`, `date`)

### 3.4 market_meta
Columns:
- `ticker` (TEXT PRIMARY KEY)
- `first_date` (TEXT)
- `last_date` (TEXT)
- `source` (TEXT)
- `notes` (TEXT)

---

### 3.5 econ_series
Columns:
- `code` (TEXT PRIMARY KEY)
- `human_name` (TEXT)
- `frequency` (TEXT)  # e.g., daily, monthly, quarterly
- `source` (TEXT)
- `notes` (TEXT)

### 3.6 econ_values
Columns:
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `code` (TEXT, indexed, FK → econ_series.code)
- `date` (TEXT, ISO `YYYY-MM-DD`, indexed)
- `revision_asof` (TEXT, ISO date, indexed)
- `value_raw` (REAL)
- `value_yoy` (REAL, nullable)
- `value_mom` (REAL, nullable)
- `value_z` (REAL, nullable)
- `value_log` (REAL, nullable)

Unique constraint:
- (`code`, `date`, `revision_asof`)

### 3.7 econ_meta
Columns:
- `code` (TEXT PRIMARY KEY)
- `last_fetched` (TEXT)
- `last_revision_asof` (TEXT)

## 4. Migrations
Create migration scripts in `db/migrations/`:

- `001_split_market_econ.sql`  
  - If legacy tables exist, move/rename or copy as needed.
- `002_create_market_tables.sql`  
  - Create `market_prices`, `market_dividends`, `market_tri`, `market_meta`.
- `003_create_economic_tables.sql`  
  - Create `econ_series`, `econ_values`, `econ_meta`.
- `004_add_multi_column_support.sql`  
  - Add multi-column fields (e.g., `value_yoy`, `value_z`) if not already present.
- `005_enforce_date_normalization.sql`  
  - Add NOT NULL constraints or checks for date fields where appropriate.
- `006_add_metadata_tables.sql`  
  - Any additional metadata tables if needed.

## 5. Integration
- Update `utils/db_connector.py` to:
  - Run pending migrations on startup or on demand.
  - Provide helpers for inserting/updating records used by fetch scripts.

## 6. Out of Scope
- Computing TRI itself (will be completed in PR #5 or integrated stepwise).
- Panel building.
- Engine integration.

## 7. Acceptance Criteria
- Database can be initialized from scratch via migrations.
- All tables exist with the schemas above.
- Basic insert/select tests pass for market and econ data.
