# PRISM SQL Database

SQLite-based storage for indicators and time series data.

## Key Concepts

### System (Domain)

**`system`**: Top-level domain for an indicator (e.g., `finance`, `climate`, `chemistry`).
A single database can store multiple systems side by side.

Valid system types:
- `finance` - Financial markets, equities, bonds, commodities
- `climate` - Climate and environmental indicators
- `chemistry` - Chemical process data
- `anthropology` - Demographic and social indicators
- `biology` - Biological and health data
- `physics` - Physical measurements and phenomena

> **Note**: Older code may reference `panel` instead of `system`. The `panel` parameter
> is deprecated and will be removed in a future version. Always use `system` for new code.

### Schema Overview

```
indicators          indicator_values
+------------+      +----------------+
| id         |----->| indicator_id   |
| name       |      | date           |
| system     |      | value          |
| frequency  |      | created_at     |
| source     |      +----------------+
| units      |
| description|
| created_at |
| updated_at |
+------------+
```

## Quick Start

### Python API

```python
from data.sql.prism_db import (
    init_db,
    add_indicator,
    write_dataframe,
    load_indicator,
    list_indicators,
)

import pandas as pd

# Initialize database (creates tables if needed)
init_db()

# Add an indicator
add_indicator(
    name="SPY",
    system="finance",
    frequency="daily",
    source="Yahoo Finance",
    units="USD",
    description="S&P 500 ETF",
)

# Write data
df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "value": [100.0, 101.5, 102.0],
})
write_dataframe(df, indicator_name="SPY", system="finance")

# Load data
data = load_indicator("SPY")
print(data)

# List all finance indicators
finance_indicators = list_indicators(system="finance")
print(finance_indicators)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PRISM_DB` | Path to the SQLite database file | `data/sql/prism.db` |

## API Reference

### Indicator Management

#### `add_indicator()`
Add a new indicator to the database.

```python
add_indicator(
    name="GDP",
    system="finance",      # Required: finance, climate, etc.
    frequency="quarterly", # daily, weekly, monthly, quarterly, yearly
    source="FRED",
    units="billions USD",
    description="Gross Domestic Product",
)
```

#### `get_indicator(name)`
Get indicator metadata by name.

```python
indicator = get_indicator("SPY")
# Returns: {'id': 1, 'name': 'SPY', 'system': 'finance', ...}
```

#### `list_indicators(system=None)`
List all indicators, optionally filtered by system.

```python
# All indicators
all_indicators = list_indicators()

# Only finance indicators
finance = list_indicators(system="finance")
```

#### `update_indicator(name, **kwargs)`
Update indicator metadata.

```python
update_indicator("SPY", description="Updated description")
```

#### `delete_indicator(name)`
Delete an indicator and all its values.

```python
delete_indicator("OLD_INDICATOR")
```

### Data Management

#### `write_dataframe()`
Write a DataFrame to the database.

```python
write_dataframe(
    df,
    indicator_name="SPY",
    system="finance",        # Required if creating new indicator
    date_column="date",      # Column name for dates (default: "date")
    value_column="value",    # Column name for values (default: "value")
    create_if_missing=True,  # Create indicator if it doesn't exist
)
```

#### `load_indicator()`
Load indicator data as a DataFrame.

```python
# Load all data
data = load_indicator("SPY")

# Load date range
data = load_indicator("SPY", start_date="2024-01-01", end_date="2024-12-31")
```

#### `load_multiple_indicators()`
Load multiple indicators into a single DataFrame.

```python
data = load_multiple_indicators(["SPY", "QQQ", "IWM"])
# Returns DataFrame with columns: SPY, QQQ, IWM
```

#### `load_system_indicators()`
Load all indicators for a given system.

```python
finance_data = load_system_indicators(
    system="finance",
    start_date="2024-01-01",
)
```

## Migrations

Migration scripts are stored in `data/sql/migrations/`.

### Running Migrations

```python
from pathlib import Path
from data.sql.prism_db import run_migration

migration_path = Path("data/sql/migrations/001_rename_panel_to_system.sql")
run_migration(migration_path)
```

### Available Migrations

| Migration | Description |
|-----------|-------------|
| `001_rename_panel_to_system.sql` | Renames `panel` column to `system` in indicators table |

## Schema Details

### `indicators` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `name` | TEXT | Unique identifier for the indicator |
| `system` | TEXT | Domain type (finance, climate, etc.) |
| `frequency` | TEXT | Data frequency (daily, weekly, etc.) |
| `source` | TEXT | Data source (FRED, Yahoo, etc.) |
| `units` | TEXT | Unit of measurement |
| `description` | TEXT | Human-readable description |
| `created_at` | TEXT | Timestamp of creation |
| `updated_at` | TEXT | Timestamp of last update |

### `indicator_values` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `indicator_id` | INTEGER | Foreign key to indicators |
| `date` | TEXT | Date in ISO format (YYYY-MM-DD) |
| `value` | REAL | Numeric value |
| `created_at` | TEXT | Timestamp of creation |

### Indexes

- `idx_indicators_system` - Fast lookups by system
- `idx_indicators_name` - Fast lookups by name
- `idx_indicator_values_date` - Fast date range queries
- `idx_indicator_values_indicator_id` - Fast lookups by indicator

## Deprecation Notice

The `panel` parameter in Python functions is **deprecated**. Use `system` instead.

```python
# OLD (deprecated - will warn)
add_indicator("SPY", panel="finance", ...)

# NEW (recommended)
add_indicator("SPY", system="finance", ...)
```

The `panel` parameter will continue to work but will emit a `DeprecationWarning`.
It will be removed in a future version.
