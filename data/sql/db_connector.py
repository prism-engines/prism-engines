"""
db_connector.py
Modern indicator-management API for PRISM Engine.

This module provides:
  - Connection management (wraps prism_db)
  - Indicator registry: add/get/list
  - Fetch logging
  - Database statistics
  - Bulk loading utilities
  - Data IO wrappers

db.py imports ONLY from this module to avoid direct prism_db dependencies.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd

from .db_path import get_db_path


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

def init_database() -> None:
    """
    Initialize the database with all required tables.
    Creates: indicators, market_prices, econ_values, fetch_log
    """
    conn = get_connection()

    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL;")

    # Create indicators table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            system TEXT DEFAULT 'market',
            source TEXT,
            frequency TEXT DEFAULT 'daily',
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create market_prices table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL,
            UNIQUE(ticker, date)
        )
    """)

    # Create econ_values table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS econ_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL,
            UNIQUE(series_id, date)
        )
    """)

    # Create fetch_log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            entity TEXT,
            operation TEXT,
            status TEXT,
            rows_fetched INTEGER DEFAULT 0,
            rows_inserted INTEGER DEFAULT 0,
            rows_updated INTEGER DEFAULT 0,
            error_message TEXT,
            started_at TEXT,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration_ms INTEGER
        )
    """)

    # Create indexes for performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market_prices_ticker ON market_prices(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market_prices_date ON market_prices(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_econ_values_series ON econ_values(series_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_econ_values_date ON econ_values(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_indicators_system ON indicators(system)")

    conn.commit()
    conn.close()


# =============================================================================
# INDICATOR REGISTRY
# =============================================================================

def add_indicator(
    name: str,
    system: str = "market",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create an indicator entry in the indicators table.

    Args:
        name: Unique identifier for the indicator
        system: Category system ('market', 'econ', etc.)
        metadata: Optional JSON-serializable metadata dict
    """
    meta_json = json.dumps(metadata) if metadata else None

    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO indicators (name, system, metadata, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (name, system, meta_json),
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
        system: Filter by system type (e.g., 'market', 'econ')

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
    source: str,
    entity: str,
    operation: str,
    status: str,
    *,
    rows_fetched: int = 0,
    rows_inserted: int = 0,
    rows_updated: int = 0,
    error_message: Optional[str] = None,
    started_at: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> int:
    """
    Log a fetch operation in the fetch_log table.

    Args:
        source: Data source (e.g., 'yahoo', 'fred')
        entity: Entity fetched (e.g., ticker symbol)
        operation: Operation type ('fetch', 'update', etc.)
        status: Result status ('success', 'error', etc.)
        rows_fetched: Number of rows fetched
        rows_inserted: Number of rows inserted
        rows_updated: Number of rows updated
        error_message: Error message if status is 'error'
        started_at: ISO timestamp when operation started
        duration_ms: Duration in milliseconds

    Returns:
        ID of the inserted log entry, or -1 if table doesn't exist
    """
    if started_at is None:
        started_at = datetime.now().isoformat()

    conn = get_connection()

    # Ensure table exists
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_log'"
    ).fetchone()

    if exists is None:
        conn.close()
        return -1

    cursor = conn.execute(
        """
        INSERT INTO fetch_log
            (source, entity, operation, status,
             rows_fetched, rows_inserted, rows_updated,
             error_message, started_at, completed_at, duration_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        """,
        (
            source,
            entity,
            operation,
            status,
            rows_fetched,
            rows_inserted,
            rows_updated,
            error_message,
            started_at,
            duration_ms,
        ),
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


def get_date_range(table: str, indicator: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Get the date range for data in a table, optionally filtered by indicator.

    Args:
        table: Table name ('market_prices' or 'econ_values')
        indicator: Optional indicator name to filter by

    Returns:
        Dictionary with 'min_date' and 'max_date' keys
    """
    conn = get_connection()

    # Determine the indicator column based on table
    if table == "market_prices":
        id_col = "ticker"
    elif table == "econ_values":
        id_col = "series_id"
    else:
        id_col = None

    if indicator and id_col:
        result = conn.execute(
            f"SELECT MIN(date), MAX(date) FROM [{table}] WHERE [{id_col}] = ?",
            (indicator,)
        ).fetchone()
    else:
        result = conn.execute(f"SELECT MIN(date), MAX(date) FROM [{table}]").fetchone()

    conn.close()

    return {
        "min_date": result[0] if result else None,
        "max_date": result[1] if result else None,
    }


# =============================================================================
# DATA IO
# =============================================================================

def write_dataframe(df: pd.DataFrame, table: str) -> int:
    """
    Write a DataFrame into a SQL table.

    Required columns:
      - market_prices: ticker, date, value
      - econ_values: series_id, date, value

    Args:
        df: DataFrame to write
        table: Target table name

    Returns:
        Number of rows written
    """
    if df.empty:
        return 0

    conn = get_connection()

    # Ensure date column is string
    df_copy = df.copy()
    if "date" in df_copy.columns:
        df_copy["date"] = df_copy["date"].astype(str)

    rows_before = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    df_copy.to_sql(table, conn, if_exists="append", index=False)
    rows_after = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]

    conn.commit()
    conn.close()

    return rows_after - rows_before


def load_indicator(name: str) -> pd.DataFrame:
    """
    Load data from market_prices or econ_values by indicator name.

    Args:
        name: Indicator name (ticker or series_id)

    Returns:
        DataFrame with columns: indicator, date, value
    """
    conn = get_connection()

    query = """
        SELECT ticker AS indicator, date, value
        FROM market_prices
        WHERE ticker = ?

        UNION ALL

        SELECT series_id AS indicator, date, value
        FROM econ_values
        WHERE series_id = ?

        ORDER BY date ASC
    """

    df = pd.read_sql(query, conn, params=[name, name])
    conn.close()
    return df


def load_multiple_indicators(names: List[str]) -> pd.DataFrame:
    """
    Load multiple indicators into a single DataFrame.

    Args:
        names: List of indicator names

    Returns:
        DataFrame with columns: indicator, date, value
    """
    if not names:
        return pd.DataFrame()

    placeholders = ",".join("?" for _ in names)

    sql = f"""
        SELECT ticker AS indicator, date, value
        FROM market_prices
        WHERE ticker IN ({placeholders})

        UNION ALL

        SELECT series_id AS indicator, date, value
        FROM econ_values
        WHERE series_id IN ({placeholders})

        ORDER BY indicator, date ASC
    """

    conn = get_connection()
    df = pd.read_sql(sql, conn, params=names + names)
    conn.close()

    return df


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
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
    df.to_csv(filepath, index=False)
    conn.close()
    return len(df)


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
    "query",
    "export_to_csv",
]
