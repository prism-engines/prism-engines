# PR #3 – Prism Fetch & Normalization Layer

## 1. Purpose
Implement robust, registry-driven data fetching for **market** and **economic** series with strict normalization and validation before any DB insertion. This is the primary defense against “garbage in → garbage out.”

## 2. Scope
- Implement:
  - `fetch_market_data.py`
  - `fetch_economic_data.py`
- Implement common normalization utilities in `utils/`.
- Fetch behavior must be **driven entirely by the registries**.

## 3. Files to Create / Modify

### 3.1 data_fetch/fetch_market_data.py
Responsibilities:
- Load `system_registry.json` and `market_registry.json`.
- For each enabled ticker:
  - Fetch raw data (e.g., from Yahoo or specified source).
  - Normalize columns:
    - `date` → ISO `YYYY-MM-DD`
    - `price`
    - `dividend`
  - Enforce:
    - Sorted dates (ascending).
    - No duplicate dates.
    - No footer garbage.
    - No invalid or future dates.
  - Optionally compute daily returns (price-based) for later use (but TRI is PR #4).
- Pass cleaned DataFrame to DB writer functions.

### 3.2 data_fetch/fetch_economic_data.py
Responsibilities:
- Load `system_registry.json` and `economic_registry.json`.
- For each enabled series:
  - Fetch raw economic time-series from its configured source.
  - Normalize columns:
    - `date` → ISO `YYYY-MM-DD`
    - `value_raw`
  - Enforce:
    - Sorted dates.
    - No duplicate dates per (code, date, revision_asof) combination.
    - Handling of revisions if enabled.
  - Compute basic derived fields if trivial (e.g., yoy/mom) **or leave for PR #4 if preferred**.
- Pass cleaned DataFrame to DB writer functions.

### 3.3 utils/date_cleaner.py
Implement:
- `parse_date_strict(str) -> datetime/date`
- Helpers to:
  - Fix 2-digit year issues.
  - Strip times and timezones.
  - Reject invalid dates.

### 3.4 utils/number_cleaner.py
Implement:
- String-to-float conversion that:
  - Strips commas, percent signs, spaces.
  - Handles common “N/A”, “null”, etc.
  - Returns `None`/NaN if not convertible.

### 3.5 utils/fetch_validator.py
Extend:
- Add functions for:
  - Validating shape (min rows, no duplicate dates, correct types).
  - Detecting and removing trailing junk (footers, text rows).

## 4. Normalization Rules (Non-Negotiable)
For all fetched series:
1. Dates must be ISO strings `YYYY-MM-DD`.
2. No duplicate date rows per series.
3. No rows that lack valid dates.
4. No forward dates beyond current date.
5. Numeric columns must be floats (or NaN).
6. Footers or non-numeric rows must be dropped.
7. Frequency should roughly match registry specification (daily/monthly/etc.); if not, log a warning.

## 5. Out of Scope
- Database table creation (migrations are PR #4).
- Multi-column DB support beyond raw fields.
- Panel-building logic.

## 6. Acceptance Criteria
- Running `fetch_market_data.py` and `fetch_economic_data.py` completes without errors for a small set of registry entries.
- Logs indicate any dropped rows, invalid values, or inconsistencies.
- DataFrames passed to the DB layer are clean, well-typed, and shape-consistent.
