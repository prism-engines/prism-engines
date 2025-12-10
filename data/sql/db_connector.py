"""
db_connector.py
Modern indicator-management API for PRISM Engine.

Schema v2 - Panel-Agnostic Architecture
========================================
All data flows through the universal `indicator_values` table.
Domain-specific tables (market_prices, econ_values) are DEPRECATED.

This module provides:
  - Connection management
  - Indicator registry: add/get/list
  - Data IO: write_dataframe, load_indicator (using indicator_values)
  - Fetch logging
  - Database statistics
  - Migration utilities

db.py imports ONLY from this module to avoid direct prism_db dependencies.
"""

import sqlite3
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
import pandas as pd

from .db_path import get_db_path


# =============================================================================
# DEPRECATED TABLE WARNINGS
# =============================================================================

_DEPRECATED_TABLES = {'market_prices', 'econ_values', 'timeseries'}

def _warn_deprecated_table(table: str) -> None:
    """Issue deprecation warning for old tables."""
    if table in _DEPRECATED_TABLES:
        warnings.warn(
            f"Table '{table}' is DEPRECATED. Use 'indicator_values' instead. "
            f"This table will be removed in a future version.",
            DeprecationWarning,
            stacklevel=3
        )


# =============================================================================
# CONNECTION MANAGEMENT
# =============================================================================

def get_connection() -> sqlite3.Connection:
    """
    Return a SQLite connection to the Prism DB.
    Sets row_factory to sqlite3.Row for dict-like access.
    """
    import os
    path = get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def connect() -> sqlite3.Connection:
    """Alias for get_connection()."""
    return get_connection()


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def _get_sql_dir() -> Path:
    """Return the directory containing SQL files."""
    return Path(__file__).parent


def _execute_schema(conn: sqlite3.Connection) -> None:
    """Execute the main schema.sql file."""
    schema_path = _get_sql_dir() / "schema.sql"
    if schema_path.exists():
        with open(schema_path, "r") as f:
            conn.executescript(f.read())
        print(f"  + Executed schema.sql")
    else:
        raise FileNotFoundError(f"schema.sql not found at {schema_path}")


def _run_migrations(conn: sqlite3.Connection) -> None:
    """
    Execute all SQL migrations in sorted order.
    Idempotent: safe to re-run even if schema elements already exist.
    """
    migrations_dir = _get_sql_dir() / "migrations"
    if not migrations_dir.exists():
        print("  - No migrations directory found")
        return

    migration_files = sorted(migrations_dir.glob("*.sql"))

    for migration_file in migration_files:
        with open(migration_file, "r") as f:
            sql_content = f.read()

        statements = [s.strip() for s in sql_content.split(";") if s.strip()]

        for statement in statements:
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as e:
                msg = str(e).lower()

                # Allow benign idempotent errors
                if (
                    "already exists" in msg
                    or "duplicate column name" in msg
                    or "cannot add a column" in msg
                ):
                    pass
                else:
                    raise

        print(f"  + Executed migration: {migration_file.name}")


def init_database() -> None:
    """
    Initialize the database by executing schema.sql and all migrations.

    Creates tables (Schema v2 - Panel-Agnostic):
      - systems (domain registry)
      - indicators (indicator metadata)
      - indicator_values (THE universal time series table)
      - fetch_log (fetch operation logs)
      - metadata (key-value store)
      - market_prices (DEPRECATED - backward compatibility)
      - econ_values (DEPRECATED - backward compatibility)
    """
    import os

    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    print(f"Initializing database at: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")

        # Execute main schema
        _execute_schema(conn)

        # Run all migrations
        _run_migrations(conn)

        conn.commit()
        print("Database initialized successfully!")

    finally:
        conn.close()


# =============================================================================
# INDICATOR REGISTRY
# =============================================================================

def add_indicator(
    name: str,
    system: str = "market",
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    frequency: str = "daily",
    description: Optional[str] = None,
    units: Optional[str] = None
) -> None:
    """
    Create or update an indicator entry in the indicators table.

    Args:
        name: Unique identifier for the indicator (e.g., 'SPY', 'GDPC1', 'co2_level')
        system: Domain system ('market', 'economic', 'climate', etc.)
        metadata: Optional JSON-serializable metadata dict
        source: Data source (e.g., 'fred', 'yahoo', 'noaa')
        frequency: Data frequency ('daily', 'weekly', 'monthly', 'quarterly')
        description: Human-readable description
        units: Units of measurement
    """
    meta_json = json.dumps(metadata) if metadata else None

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO indicators (name, system, metadata, source, frequency, description, units, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(name) DO UPDATE SET
            system = excluded.system,
            metadata = excluded.metadata,
            source = excluded.source,
            frequency = excluded.frequency,
            description = excluded.description,
            units = excluded.units,
            updated_at = CURRENT_TIMESTAMP
        """,
        (name, system, meta_json, source, frequency, description, units),
    )
    conn.commit()
    conn.close()


def get_indicator(name: str) -> Optional[Dict[str, Any]]:
    """
    Return indicator metadata as a dictionary.
    Returns None if indicator doesn't exist.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM indicators WHERE name = ?", (name,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    result = dict(row)
    # Parse metadata JSON if present
    if result.get("metadata"):
        try:
            result["metadata"] = json.loads(result["metadata"])
        except (json.JSONDecodeError, TypeError):
            pass

    return result


def list_indicators(system: Optional[str] = None) -> List[str]:
    """
    List indicator names, optionally filtered by system.

    Args:
        system: Filter by system type (e.g., 'market', 'economic', 'climate')

    Returns:
        List of indicator names
    """
    conn = get_connection()

    if system:
        rows = conn.execute(
            "SELECT name FROM indicators WHERE system = ? ORDER BY name",
            (system,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name FROM indicators ORDER BY name"
        ).fetchall()

    conn.close()
    return [r[0] for r in rows]


# =============================================================================
# FETCH LOGGING
# =============================================================================

def log_fetch(
    indicator_name: str,
    source: str,
    status: str,
    *,
    rows_fetched: int = 0,
    error_message: Optional[str] = None,
) -> int:
    """
    Log a fetch operation in the fetch_log table.

    Args:
        indicator_name: Name of the indicator fetched
        source: Data source (e.g., 'yahoo', 'fred', 'tiingo')
        status: Result status ('success', 'error', 'partial')
        rows_fetched: Number of rows fetched
        error_message: Error message if status is 'error'

    Returns:
        ID of the inserted log entry, or -1 if table doesn't exist
    """
    conn = get_connection()

    cursor = conn.execute(
        """
        INSERT INTO fetch_log
            (indicator_name, source, fetch_date, rows_fetched, status, error_message)
        VALUES (?, ?, datetime('now'), ?, ?, ?)
        """,
        (indicator_name, source, rows_fetched, status, error_message),
    )
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id


# =============================================================================
# DATABASE STATISTICS
# =============================================================================

def database_stats() -> Dict[str, Any]:
    """
    Return comprehensive database statistics.

    Returns:
        Dictionary with table counts and database info
    """
    conn = get_connection()

    stats = {
        "tables": {},
        "total_rows": 0,
        "db_path": str(get_db_path()),
        "schema_version": "v2",
    }

    # Get all tables
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()

    for t in tables:
        table_name = t[0]
        count = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
        stats["tables"][table_name] = count
        stats["total_rows"] += count

    conn.close()
    return stats


def get_table_stats(table: str) -> Dict[str, Any]:
    """
    Return detailed statistics for a specific table.

    Args:
        table: Name of the table

    Returns:
        Dictionary with row count, columns, and sample data info
    """
    conn = get_connection()

    # Check table exists
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()

    if not exists:
        conn.close()
        return {"error": f"Table '{table}' does not exist"}

    stats = {"table": table}

    # Row count
    stats["row_count"] = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]

    # Column info
    columns = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    stats["columns"] = [{"name": c[1], "type": c[2]} for c in columns]

    # Date range if date column exists
    col_names = [c[1] for c in columns]
    if "date" in col_names:
        result = conn.execute(f"SELECT MIN(date), MAX(date) FROM [{table}]").fetchone()
        stats["date_range"] = {"min": result[0], "max": result[1]}

    conn.close()
    return stats


def get_date_range(
    indicator_name: Optional[str] = None,
    table: str = "indicator_values"
) -> Dict[str, Optional[str]]:
    """
    Get the date range for data, optionally filtered by indicator.

    Args:
        indicator_name: Optional indicator name to filter by
        table: Table name (default: 'indicator_values')

    Returns:
        Dictionary with 'min_date' and 'max_date' keys
    """
    conn = get_connection()

    # Handle deprecated tables
    if table in _DEPRECATED_TABLES:
        _warn_deprecated_table(table)

    if table == "indicator_values":
        if indicator_name:
            result = conn.execute(
                "SELECT MIN(date), MAX(date) FROM indicator_values WHERE indicator_name = ?",
                (indicator_name,)
            ).fetchone()
        else:
            result = conn.execute("SELECT MIN(date), MAX(date) FROM indicator_values").fetchone()
    elif table == "market_prices":
        if indicator_name:
            result = conn.execute(
                "SELECT MIN(date), MAX(date) FROM market_prices WHERE ticker = ?",
                (indicator_name,)
            ).fetchone()
        else:
            result = conn.execute("SELECT MIN(date), MAX(date) FROM market_prices").fetchone()
    elif table == "econ_values":
        if indicator_name:
            result = conn.execute(
                "SELECT MIN(date), MAX(date) FROM econ_values WHERE series_id = ?",
                (indicator_name,)
            ).fetchone()
        else:
            result = conn.execute("SELECT MIN(date), MAX(date) FROM econ_values").fetchone()
    else:
        result = conn.execute(f"SELECT MIN(date), MAX(date) FROM [{table}]").fetchone()

    conn.close()

    return {
        "min_date": result[0] if result else None,
        "max_date": result[1] if result else None,
    }


# =============================================================================
# DATA IO - Schema v2 (Panel-Agnostic)
# =============================================================================

def write_dataframe(
    df: pd.DataFrame,
    table: str = "indicator_values",
    indicator_name: Optional[str] = None,
    provenance: Optional[str] = None,
    quality_flag: Optional[str] = None
) -> int:
    """
    Write a DataFrame into the database.

    Schema v2: All data should go to 'indicator_values' table.
    Legacy tables (market_prices, econ_values) are supported but deprecated.

    For indicator_values, expected columns:
      - indicator_name (or provide via parameter)
      - date
      - value
      - quality_flag (optional)
      - provenance (optional)
      - extra (optional, JSON)

    Args:
        df: DataFrame to write
        table: Target table (default: 'indicator_values')
        indicator_name: Indicator name (required for indicator_values if not in df)
        provenance: Data provenance/source (e.g., 'fred', 'yahoo')
        quality_flag: Data quality flag (e.g., 'verified', 'estimated')

    Returns:
        Number of rows written
    """
    if df.empty:
        return 0

    # Warn about deprecated tables
    if table in _DEPRECATED_TABLES:
        _warn_deprecated_table(table)

    conn = get_connection()
    df_copy = df.copy()

    # Ensure date column is string
    if "date" in df_copy.columns:
        df_copy["date"] = pd.to_datetime(df_copy["date"]).dt.strftime('%Y-%m-%d')

    # Handle indicator_values table specially
    if table == "indicator_values":
        # Add indicator_name if provided and not in df
        if indicator_name and "indicator_name" not in df_copy.columns:
            df_copy["indicator_name"] = indicator_name

        # Add provenance if provided
        if provenance and "provenance" not in df_copy.columns:
            df_copy["provenance"] = provenance

        # Add quality_flag if provided
        if quality_flag and "quality_flag" not in df_copy.columns:
            df_copy["quality_flag"] = quality_flag

        # Ensure required columns exist
        if "indicator_name" not in df_copy.columns:
            raise ValueError("indicator_name required for indicator_values table")

        # Use INSERT OR REPLACE for upsert behavior
        rows_written = 0
        for _, row in df_copy.iterrows():
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO indicator_values
                        (indicator_name, date, value, quality_flag, provenance, extra)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("indicator_name"),
                        row.get("date"),
                        row.get("value"),
                        row.get("quality_flag"),
                        row.get("provenance"),
                        row.get("extra"),
                    )
                )
                rows_written += 1
            except sqlite3.IntegrityError:
                pass  # Skip duplicates

        conn.commit()
        conn.close()
        return rows_written

    # Handle legacy tables
    rows_before = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    df_copy.to_sql(table, conn, if_exists="append", index=False)
    rows_after = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]

    conn.commit()
    conn.close()

    return rows_after - rows_before


def load_indicator(name: str, include_metadata: bool = False) -> pd.DataFrame:
    """
    Load time series data for an indicator from indicator_values.

    Falls back to legacy tables (market_prices, econ_values) if not found
    in indicator_values.

    Args:
        name: Indicator name
        include_metadata: Include quality_flag, provenance, extra columns

    Returns:
        DataFrame with columns: indicator_name, date, value, [quality_flag, provenance, extra]
    """
    conn = get_connection()

    # Try indicator_values first (Schema v2)
    if include_metadata:
        query = """
            SELECT indicator_name, date, value, quality_flag, provenance, extra
            FROM indicator_values
            WHERE indicator_name = ?
            ORDER BY date ASC
        """
    else:
        query = """
            SELECT indicator_name, date, value
            FROM indicator_values
            WHERE indicator_name = ?
            ORDER BY date ASC
        """

    df = pd.read_sql(query, conn, params=[name])

    # If found in indicator_values, return it
    if not df.empty:
        conn.close()
        return df

    # Fallback to legacy tables (with deprecation warning)
    warnings.warn(
        f"Indicator '{name}' not found in indicator_values. "
        "Falling back to deprecated tables (market_prices, econ_values). "
        "Please migrate data to indicator_values.",
        DeprecationWarning,
        stacklevel=2
    )

    # Try market_prices
    df = pd.read_sql(
        "SELECT ticker AS indicator_name, date, value FROM market_prices WHERE ticker = ? ORDER BY date ASC",
        conn, params=[name]
    )

    if not df.empty:
        conn.close()
        return df

    # Try econ_values
    df = pd.read_sql(
        "SELECT series_id AS indicator_name, date, value FROM econ_values WHERE series_id = ? ORDER BY date ASC",
        conn, params=[name]
    )

    conn.close()
    return df


def load_multiple_indicators(
    names: List[str],
    include_metadata: bool = False
) -> pd.DataFrame:
    """
    Load multiple indicators into a single DataFrame.

    Args:
        names: List of indicator names
        include_metadata: Include quality_flag, provenance, extra columns

    Returns:
        DataFrame with columns: indicator_name, date, value, [quality_flag, provenance, extra]
    """
    if not names:
        return pd.DataFrame()

    conn = get_connection()
    placeholders = ",".join("?" for _ in names)

    if include_metadata:
        query = f"""
            SELECT indicator_name, date, value, quality_flag, provenance, extra
            FROM indicator_values
            WHERE indicator_name IN ({placeholders})
            ORDER BY indicator_name, date ASC
        """
    else:
        query = f"""
            SELECT indicator_name, date, value
            FROM indicator_values
            WHERE indicator_name IN ({placeholders})
            ORDER BY indicator_name, date ASC
        """

    df = pd.read_sql(query, conn, params=names)

    # Check if any indicators are missing from indicator_values
    found_indicators = set(df['indicator_name'].unique()) if not df.empty else set()
    missing_indicators = set(names) - found_indicators

    if missing_indicators:
        # Fallback to legacy tables for missing indicators
        warnings.warn(
            f"Indicators {missing_indicators} not found in indicator_values. "
            "Falling back to deprecated tables.",
            DeprecationWarning,
            stacklevel=2
        )

        missing_placeholders = ",".join("?" for _ in missing_indicators)
        missing_list = list(missing_indicators)

        # Try market_prices
        legacy_df = pd.read_sql(
            f"SELECT ticker AS indicator_name, date, value FROM market_prices WHERE ticker IN ({missing_placeholders})",
            conn, params=missing_list
        )

        if not legacy_df.empty:
            df = pd.concat([df, legacy_df], ignore_index=True)

        # Try econ_values
        legacy_df = pd.read_sql(
            f"SELECT series_id AS indicator_name, date, value FROM econ_values WHERE series_id IN ({missing_placeholders})",
            conn, params=missing_list
        )

        if not legacy_df.empty:
            df = pd.concat([df, legacy_df], ignore_index=True)

        # Sort again after concatenation
        if not df.empty:
            df = df.sort_values(['indicator_name', 'date']).reset_index(drop=True)

    conn.close()
    return df


def load_all_indicators_wide(
    indicators: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ffill: bool = True
) -> pd.DataFrame:
    """
    Load indicators from indicator_values and pivot to wide format.

    This is the primary data loading function for analysis scripts.
    Returns a DataFrame with date as index and indicators as columns.

    Falls back to legacy tables if indicator_values is empty.

    Args:
        indicators: Optional list of indicator names to load (None = all)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        ffill: Forward-fill missing values (default True). Essential for
               monthly/quarterly economic data stored in daily format.

    Returns:
        Wide-format DataFrame with date index and indicator columns
    """
    conn = get_connection()

    # Build query
    if indicators:
        placeholders = ",".join("?" for _ in indicators)
        query = f"""
            SELECT indicator_name, date, value
            FROM indicator_values
            WHERE indicator_name IN ({placeholders})
        """
        params = list(indicators)
    else:
        query = "SELECT indicator_name, date, value FROM indicator_values WHERE 1=1"
        params = []

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date, indicator_name"

    df = pd.read_sql(query, conn, params=params)

    # Fallback to legacy tables if indicator_values is empty
    if df.empty:
        warnings.warn(
            "indicator_values is empty. Falling back to legacy tables "
            "(market_prices, econ_values). Please migrate data using migrate_legacy_tables().",
            DeprecationWarning,
            stacklevel=2
        )

        # Load from market_prices
        market_df = pd.read_sql(
            "SELECT ticker AS indicator_name, date, value FROM market_prices ORDER BY date",
            conn
        )

        # Load from econ_values
        econ_df = pd.read_sql(
            "SELECT series_id AS indicator_name, date, value FROM econ_values ORDER BY date",
            conn
        )

        df = pd.concat([market_df, econ_df], ignore_index=True)

        if not df.empty:
            # Apply filters
            if indicators:
                df = df[df['indicator_name'].isin(indicators)]
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]

    conn.close()

    if df.empty:
        return pd.DataFrame()

    # Pivot to wide format
    df['date'] = pd.to_datetime(df['date'])
    df_wide = df.pivot(index='date', columns='indicator_name', values='value')
    df_wide = df_wide.sort_index()
    
    # Forward fill missing values
    # Essential for monthly/quarterly data (CPI, GDP, etc.) stored sparsely in daily format
    # January's CPI value should persist until February's release
    if ffill:
        df_wide = df_wide.ffill()

    return df_wide


def query(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """
    Run an arbitrary SQL query and return a DataFrame.

    Args:
        sql: SQL query string
        params: Optional query parameters

    Returns:
        Query results as DataFrame
    """
    conn = get_connection()
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def export_to_csv(table: str, filepath: str) -> int:
    """
    Export a SQL table to CSV file.

    Args:
        table: Table name to export
        filepath: Destination file path

    Returns:
        Number of rows exported
    """
    if table in _DEPRECATED_TABLES:
        _warn_deprecated_table(table)

    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
    df.to_csv(filepath, index=False)
    conn.close()
    return len(df)


# =============================================================================
# MIGRATION UTILITIES
# =============================================================================

def migrate_legacy_tables(dry_run: bool = False) -> Dict[str, int]:
    """
    Migrate data from legacy tables (market_prices, econ_values) to indicator_values.

    Args:
        dry_run: If True, only report what would be migrated without actually migrating

    Returns:
        Dictionary with migration statistics
    """
    conn = get_connection()

    stats = {
        "market_prices_rows": 0,
        "econ_values_rows": 0,
        "total_migrated": 0,
        "errors": 0,
    }

    # Count rows in legacy tables
    try:
        stats["market_prices_rows"] = conn.execute(
            "SELECT COUNT(*) FROM market_prices"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        pass

    try:
        stats["econ_values_rows"] = conn.execute(
            "SELECT COUNT(*) FROM econ_values"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        pass

    if dry_run:
        print(f"DRY RUN - Would migrate:")
        print(f"  market_prices: {stats['market_prices_rows']} rows")
        print(f"  econ_values: {stats['econ_values_rows']} rows")
        conn.close()
        return stats

    # Migrate market_prices
    try:
        conn.execute("""
            INSERT OR IGNORE INTO indicator_values (indicator_name, date, value, provenance)
            SELECT ticker, date, value, 'migration:market_prices'
            FROM market_prices
            WHERE ticker IS NOT NULL AND date IS NOT NULL AND value IS NOT NULL
        """)
        stats["total_migrated"] += stats["market_prices_rows"]
        print(f"  + Migrated {stats['market_prices_rows']} rows from market_prices")
    except sqlite3.Error as e:
        print(f"  ! Error migrating market_prices: {e}")
        stats["errors"] += 1

    # Migrate econ_values
    try:
        conn.execute("""
            INSERT OR IGNORE INTO indicator_values (indicator_name, date, value, provenance)
            SELECT series_id, date, value, 'migration:econ_values'
            FROM econ_values
            WHERE series_id IS NOT NULL AND date IS NOT NULL AND value IS NOT NULL
        """)
        stats["total_migrated"] += stats["econ_values_rows"]
        print(f"  + Migrated {stats['econ_values_rows']} rows from econ_values")
    except sqlite3.Error as e:
        print(f"  ! Error migrating econ_values: {e}")
        stats["errors"] += 1

    conn.commit()
    conn.close()

    return stats


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Connection
    "get_connection",
    "connect",
    # Initialization
    "init_database",
    # Indicator registry
    "add_indicator",
    "get_indicator",
    "list_indicators",
    # Fetch logging
    "log_fetch",
    # Statistics
    "database_stats",
    "get_table_stats",
    "get_date_range",
    # Data IO
    "write_dataframe",
    "load_indicator",
    "load_multiple_indicators",
    "load_all_indicators_wide",
    "query",
    "export_to_csv",
    # Migration
    "migrate_legacy_tables",
    # Analysis results
    "init_analysis_tables",
    "save_analysis_run",
    "get_analysis_runs",
    "get_run_results",
    "get_indicator_rankings",
]


# =============================================================================
# ANALYSIS RESULTS STORAGE
# =============================================================================

def init_analysis_tables() -> None:
    """
    Create tables for storing analysis results.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    conn = get_connection()
    
    conn.executescript("""
        -- Analysis run metadata
        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            start_date TEXT,
            end_date TEXT,
            n_indicators INTEGER,
            n_rows INTEGER,
            n_lenses INTEGER,
            n_errors INTEGER,
            config TEXT,  -- JSON
            runtime_seconds REAL
        );
        
        -- Per-lens results
        CREATE TABLE IF NOT EXISTS lens_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES analysis_runs(run_id),
            lens_name TEXT NOT NULL,
            normalize_method TEXT,
            result TEXT,  -- JSON
            runtime_seconds REAL,
            error TEXT,
            UNIQUE(run_id, lens_name)
        );
        
        -- Indicator rankings (denormalized for easy querying)
        CREATE TABLE IF NOT EXISTS indicator_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES analysis_runs(run_id),
            indicator TEXT NOT NULL,
            lens_name TEXT NOT NULL,
            score REAL,
            rank INTEGER,
            UNIQUE(run_id, indicator, lens_name)
        );
        
        -- Consensus rankings per run
        CREATE TABLE IF NOT EXISTS consensus_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER REFERENCES analysis_runs(run_id),
            indicator TEXT NOT NULL,
            mean_score REAL,
            n_lenses INTEGER,
            rank INTEGER,
            UNIQUE(run_id, indicator)
        );
        
        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_lens_results_run ON lens_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_indicator_rankings_run ON indicator_rankings(run_id);
        CREATE INDEX IF NOT EXISTS idx_indicator_rankings_indicator ON indicator_rankings(indicator);
        CREATE INDEX IF NOT EXISTS idx_consensus_rankings_run ON consensus_rankings(run_id);
    """)
    
    conn.commit()
    conn.close()


def save_analysis_run(
    start_date: str,
    end_date: Optional[str],
    n_indicators: int,
    n_rows: int,
    n_lenses: int,
    n_errors: int,
    config: Optional[Dict] = None,
    runtime_seconds: Optional[float] = None,
    lens_results: Optional[Dict[str, Dict]] = None,
    lens_errors: Optional[Dict[str, str]] = None,
    rankings: Optional[Dict[str, pd.DataFrame]] = None,
    consensus: Optional[pd.DataFrame] = None,
    normalize_methods: Optional[Dict[str, str]] = None,
) -> int:
    """
    Save a complete analysis run to the database.
    
    Args:
        start_date: Analysis start date
        end_date: Analysis end date (None = today)
        n_indicators: Number of indicators analyzed
        n_rows: Number of data rows
        n_lenses: Number of lenses run
        n_errors: Number of lens errors
        config: Configuration dict (will be JSON serialized)
        runtime_seconds: Total runtime
        lens_results: Dict of lens_name -> result dict
        lens_errors: Dict of lens_name -> error string
        rankings: Dict of lens_name -> DataFrame with indicator rankings
        consensus: DataFrame with consensus rankings
        normalize_methods: Dict of lens_name -> normalization method used
    
    Returns:
        run_id of the saved run
    """
    import json
    
    conn = get_connection()
    
    # Ensure tables exist
    init_analysis_tables()
    
    # Insert run metadata
    cursor = conn.execute(
        """
        INSERT INTO analysis_runs 
            (start_date, end_date, n_indicators, n_rows, n_lenses, n_errors, config, runtime_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            start_date,
            end_date,
            n_indicators,
            n_rows,
            n_lenses,
            n_errors,
            json.dumps(config) if config else None,
            runtime_seconds
        )
    )
    run_id = cursor.lastrowid
    
    # Helper to make objects JSON-serializable
    def clean_for_json(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.float64, np.int32, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj
    
    # Insert lens results
    if lens_results:
        for lens_name, result in lens_results.items():
            norm_method = normalize_methods.get(lens_name) if normalize_methods else None
            conn.execute(
                """
                INSERT OR REPLACE INTO lens_results 
                    (run_id, lens_name, normalize_method, result, error)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    lens_name,
                    norm_method,
                    json.dumps(clean_for_json(result)),
                    None
                )
            )
    
    # Insert lens errors
    if lens_errors:
        for lens_name, error in lens_errors.items():
            norm_method = normalize_methods.get(lens_name) if normalize_methods else None
            conn.execute(
                """
                INSERT OR REPLACE INTO lens_results 
                    (run_id, lens_name, normalize_method, result, error)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    lens_name,
                    norm_method,
                    None,
                    str(error)[:500]  # Truncate long errors
                )
            )
    
    # Insert per-lens rankings
    if rankings:
        for lens_name, ranking_df in rankings.items():
            if isinstance(ranking_df, pd.DataFrame) and not ranking_df.empty:
                # Make a copy to avoid modifying original
                ranking_df = ranking_df.copy()
                
                # Add rank column if not present
                if 'score' in ranking_df.columns:
                    ranking_df = ranking_df.sort_values('score', ascending=False)
                    ranking_df['rank'] = range(1, len(ranking_df) + 1)
                    
                    for idx, row in ranking_df.iterrows():
                        # Try to get indicator name from multiple sources
                        if 'indicator' in ranking_df.columns:
                            indicator = str(row['indicator'])
                        elif isinstance(idx, str) and not idx.isdigit():
                            indicator = idx
                        elif hasattr(ranking_df.index, 'name') and ranking_df.index.name:
                            indicator = str(idx)
                        else:
                            # Skip if we can't determine indicator name
                            continue
                        
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO indicator_rankings
                                (run_id, indicator, lens_name, score, rank)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                run_id,
                                indicator,
                                lens_name,
                                float(row.get('score', 0)),
                                int(row.get('rank', 0))
                            )
                        )
    
    # Insert consensus rankings
    if consensus is not None and not consensus.empty:
        consensus = consensus.sort_values('mean_score', ascending=False)
        for rank, (indicator, row) in enumerate(consensus.iterrows(), 1):
            conn.execute(
                """
                INSERT OR REPLACE INTO consensus_rankings
                    (run_id, indicator, mean_score, n_lenses, rank)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(indicator),
                    float(row.get('mean_score', 0)),
                    int(row.get('n_lenses', 0)),
                    rank
                )
            )
    
    conn.commit()
    conn.close()
    
    return run_id


def get_analysis_runs(limit: int = 10) -> pd.DataFrame:
    """
    Get recent analysis runs.
    
    Args:
        limit: Max number of runs to return
    
    Returns:
        DataFrame with run metadata
    """
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT run_id, run_date, start_date, end_date, 
               n_indicators, n_rows, n_lenses, n_errors, runtime_seconds
        FROM analysis_runs
        ORDER BY run_date DESC
        LIMIT ?
        """,
        conn,
        params=(limit,)
    )
    conn.close()
    return df


def get_run_results(run_id: int) -> Dict[str, Any]:
    """
    Get full results for a specific run.
    
    Args:
        run_id: The run ID to retrieve
    
    Returns:
        Dict with run metadata, lens results, and rankings
    """
    import json
    
    conn = get_connection()
    
    # Get run metadata
    run_row = conn.execute(
        "SELECT * FROM analysis_runs WHERE run_id = ?",
        (run_id,)
    ).fetchone()
    
    if not run_row:
        conn.close()
        return {}
    
    result = dict(run_row)
    if result.get('config'):
        try:
            result['config'] = json.loads(result['config'])
        except:
            pass
    
    # Get lens results
    lens_df = pd.read_sql(
        "SELECT lens_name, normalize_method, result, error FROM lens_results WHERE run_id = ?",
        conn,
        params=(run_id,)
    )
    result['lens_results'] = {}
    result['lens_errors'] = {}
    for _, row in lens_df.iterrows():
        if row['error']:
            result['lens_errors'][row['lens_name']] = row['error']
        elif row['result']:
            try:
                result['lens_results'][row['lens_name']] = json.loads(row['result'])
            except:
                pass
    
    # Get consensus rankings
    consensus_df = pd.read_sql(
        """
        SELECT indicator, mean_score, n_lenses, rank
        FROM consensus_rankings
        WHERE run_id = ?
        ORDER BY rank
        """,
        conn,
        params=(run_id,)
    )
    result['consensus'] = consensus_df
    
    conn.close()
    return result


def get_indicator_rankings(
    indicator: str,
    limit: int = 10
) -> pd.DataFrame:
    """
    Get ranking history for a specific indicator across runs.
    
    Args:
        indicator: Indicator name to look up
        limit: Max number of runs to return
    
    Returns:
        DataFrame with ranking history
    """
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT 
            ar.run_id,
            ar.run_date,
            ar.start_date,
            ar.end_date,
            cr.mean_score,
            cr.n_lenses,
            cr.rank
        FROM consensus_rankings cr
        JOIN analysis_runs ar ON cr.run_id = ar.run_id
        WHERE cr.indicator = ?
        ORDER BY ar.run_date DESC
        LIMIT ?
        """,
        conn,
        params=(indicator, limit)
    )
    conn.close()
    return df
