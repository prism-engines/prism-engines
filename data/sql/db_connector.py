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
    """Run all SQL migrations in numeric order."""
    migrations_dir = _get_sql_dir() / "migrations"
    if not migrations_dir.exists():
        print(f"  - No migrations directory found")
        return

    # Get all .sql files sorted by name (numeric prefix)
    migration_files = sorted(migrations_dir.glob("*.sql"))

    for migration_file in migration_files:
        with open(migration_file, "r") as f:
            sql_content = f.read()

        # Execute each statement separately to handle IF NOT EXISTS gracefully
        for statement in sql_content.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as e:
                    # Ignore "table already exists" errors
                    if "already exists" not in str(e):
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
    end_date: Optional[str] = None
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
]
