# PRISM SQL Data Layer

This directory contains the SQLite-based data infrastructure for PRISM Engine. All data ingestion, fetching, cleaning, and output operations route through SQLite databases for improved data integrity, querying capabilities, and workflow management.

## Overview

The SQL data layer provides:
- **Centralized storage** for all raw and cleaned data
- **Query capabilities** for efficient data retrieval
- **Data quality tracking** with metrics and cleaning history
- **Panel management** for consolidated analysis datasets

## Database Location

The primary database is located at:
```
data/sql/prism.db
```

## Schema

### Tables

#### `raw_data`
Stores original fetched data from all sources.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| date | TEXT | Date in YYYY-MM-DD format |
| ticker | TEXT | Ticker symbol (lowercase) |
| source | TEXT | Data source ('yahoo', 'fred', 'climate', 'custom') |
| value | REAL | Primary value |
| open | REAL | Open price (for market data) |
| high | REAL | High price |
| low | REAL | Low price |
| close | REAL | Close price |
| volume | REAL | Trading volume |
| created_at | TIMESTAMP | Record creation time |

#### `data_sources`
Metadata about data sources and fetch operations.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| ticker | TEXT | Ticker symbol (unique) |
| source | TEXT | Data source |
| name | TEXT | Human-readable name |
| category | TEXT | Category ('equity', 'macro', 'rates', etc.) |
| description | TEXT | Optional description |
| first_date | TEXT | First available date |
| last_date | TEXT | Last available date |
| row_count | INTEGER | Number of data points |
| last_fetched | TIMESTAMP | Last fetch time |

#### `cleaned_data`
Processed data after cleaning pipeline.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| date | TEXT | Date in YYYY-MM-DD format |
| ticker | TEXT | Ticker symbol |
| value | REAL | Cleaned value |
| cleaning_method | TEXT | Method used ('ffill', 'interpolate', etc.) |
| created_at | TIMESTAMP | Record creation time |

#### `data_quality`
Data quality metrics after cleaning.

| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT | Ticker symbol (unique) |
| source_rows | INTEGER | Original row count |
| cleaned_rows | INTEGER | Cleaned row count |
| nan_count_before | INTEGER | NaN count before cleaning |
| nan_count_after | INTEGER | NaN count after cleaning |
| nan_pct_before | REAL | NaN percentage before |
| nan_pct_after | REAL | NaN percentage after |
| outliers_detected | INTEGER | Outliers found |
| outliers_handled | INTEGER | Outliers processed |
| cleaning_method | TEXT | Cleaning method used |

#### `panels`
Consolidated panel metadata.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | Panel name (unique) |
| description | TEXT | Optional description |
| start_date | TEXT | Panel start date |
| end_date | TEXT | Panel end date |
| n_indicators | INTEGER | Number of indicators |
| n_rows | INTEGER | Number of rows |
| indicators | TEXT | JSON array of indicator names |

#### `panel_data`
Panel data in long format.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| panel_id | INTEGER | Foreign key to panels |
| date | TEXT | Date |
| ticker | TEXT | Ticker symbol |
| value | REAL | Value |

## Usage

### Python API

```python
from data.sql import SQLDataManager

# Initialize database
db = SQLDataManager()
db.init_schema()

# Store raw data from a fetch
db.store_raw_data(df, source='yahoo', ticker='SPY', category='equity')

# Store cleaned data
db.store_cleaned_data(df, ticker='SPY', cleaning_method='ffill')

# Load data as a panel (wide format)
panel = db.load_cleaned_panel()

# Load specific indicators
panel = db.load_cleaned_panel(tickers=['spy', 'qqq', 'gld'])

# Query single indicator
spy_data = db.query_indicator('spy')

# Get data quality report
quality_df = db.get_data_quality_report()

# Get database statistics
stats = db.get_stats()
```

### Convenience Functions

```python
from data.sql import store_dataframe, load_dataframe, query_indicator

# Store data (automatically handles raw vs cleaned)
store_dataframe(df, source='yahoo', ticker='SPY')
store_dataframe(df, source='yahoo', ticker='SPY', cleaned=True)

# Load data
panel = load_dataframe(as_panel=True)
indicator = load_dataframe(ticker='spy')

# Query specific indicator
spy = query_indicator('spy')
```

### Integration with Fetcher

The `fetcher.py` has been updated to optionally store data in SQL:

```python
from fetcher import fetch_all

# Fetch and store in SQL
panel = fetch_all(save_to_sql=True)
```

### Loading for Analysis

In `main.py` and other analysis scripts:

```python
from data.sql import SQLDataManager

db = SQLDataManager()
panel = db.load_cleaned_panel()

# Or load from CSV for backwards compatibility
panel = pd.read_csv('data/raw/master_panel.csv', index_col=0, parse_dates=True)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│   Yahoo Finance │ FRED │ Climate API │ Custom Sources          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     01_FETCH (Fetchers)                         │
│   FREDFetcher │ YahooFetcher │ ClimateFetcher │ CustomFetcher  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   data/sql/prism.db                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    raw_data                             │   │
│   │   date │ ticker │ source │ value │ open │ high │ ...   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              03_CLEANING Pipeline                       │   │
│   │   NaN handling │ Outlier detection │ Alignment          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  cleaned_data                           │   │
│   │   date │ ticker │ value │ cleaning_method               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  panels / panel_data                    │   │
│   │   Consolidated analysis-ready datasets                  │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    05_ENGINE (Analysis)                         │
│   Lenses │ Consensus │ Temporal Analysis                        │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    06_OUTPUT (Results)                          │
│   Uses TemporalDB from utils/db_manager.py for results storage │
└─────────────────────────────────────────────────────────────────┘
```

## Migration from CSV

### Importing Existing CSV Data

To migrate existing CSV data to the SQL database:

```python
from data.sql import SQLDataManager
import pandas as pd
from pathlib import Path

db = SQLDataManager()
db.init_schema()

# Import raw CSVs
raw_dir = Path('data/raw')
for csv_file in raw_dir.glob('*.csv'):
    ticker = csv_file.stem
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df = df.reset_index()
    df.columns = ['date'] + list(df.columns[1:])
    
    # Determine source
    source = 'yahoo'  # or 'fred' based on ticker
    
    db.store_raw_data(df, source=source, ticker=ticker)
```

### Exporting to CSV (if needed)

```python
# Export cleaned panel to CSV
panel = db.load_cleaned_panel()
panel.to_csv('data/raw/master_panel.csv')

# Export single indicator
spy = db.query_indicator('spy')
spy.to_csv('data/raw/spy.csv', index=False)
```

## Files in this Directory

- `__init__.py` - Module exports
- `sql_data_manager.py` - Core SQLDataManager class
- `prism.db` - SQLite database (created on first use)
- `README.md` - This documentation

## Related Modules

- `utils/db_manager.py` - TemporalDB for analysis results storage
- `fetcher.py` - Data fetching with SQL integration
- `loader.py` - Data loading utilities
- `main.py` - Main analysis pipeline

## Best Practices

1. **Always initialize the schema** before first use:
   ```python
   db = SQLDataManager()
   db.init_schema()
   ```

2. **Use lowercase tickers** - The system normalizes to lowercase internally

3. **Check data quality** after cleaning:
   ```python
   quality = db.get_data_quality_report()
   print(quality)
   ```

4. **Use panels for analysis** - Load as panel for wide-format analysis:
   ```python
   panel = db.load_cleaned_panel()
   ```

5. **Store metadata** when fetching:
   ```python
   db.store_raw_data(df, source='yahoo', ticker='SPY', 
                     name='S&P 500 ETF', category='equity')
   ```
