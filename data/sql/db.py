"""
PRISM Engine - SQLite Database Layer
------------------------------------
This module provides a clean, modern API for storing and retrieving
indicator time-series across multiple systems (finance, economic, climate, etc.).

The DB path can be configured via:
1. PRISM_DB environment variable
2. system_registry.json paths configuration
3. Default: data/sql/engine.db

Tables are created automatically from schema.sql.

Usage:
    from data.sql.db import init_database, add_indicator, write_dataframe, load_indicator

    # Initialize (creates tables if needed)
    init_database()

    # Add indicator metadata
    add_indicator('SPY', system='market', source='yahoo')

    # Write time series data
    df = pd.DataFrame({'date': ['2024-01-01'], 'value': [450.0]})
    write_dataframe(df, indicator='SPY', system='market')

    # Load data
    data = load_indicator('SPY', system='market')
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd


# ============================================================
# PATH RESOLUTION
# ============================================================

def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_db_path() -> str:
    """
    Determine where the SQLite database should live.

    Priority:
    1. Environment variable: PRISM_DB
    2. Registry configuration: paths.database.{active_db_path}
    3. Default location: data/sql/engine.db
    
    Returns:
        Absolute path to the database file
    """
    # 1. Environment variable takes precedence
    env_path = os.getenv("PRISM_DB")
    if env_path:
        return os.path.expanduser(env_path)
    
    # 2. Check registry for configured path
    root = _get_project_root()
    registry_path = root / "data" / "registry" / "system_registry.json"
    
    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
            
            paths = registry.get("paths", {})
            database_paths = paths.get("database", {})
            active_key = paths.get("active_db_path", "default")
            
            if active_key in database_paths:
                db_path = database_paths[active_key]
                return os.path.expanduser(db_path)
        except (json.JSONDecodeError, KeyError):
            pass  # Fall through to default
    
    # 3. Default location
    return str(root / "data" / "sql" / "engine.db")


# ============================================================
# CONNECTION
# ============================================================

def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Connect to the SQLite database, creating directories if needed.
    
    Args:
        db_path: Optional path to database. Uses get_db_path() if None.
        
    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row
    """
    if db_path is None:
        db_path = get_db_path()
    
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database(db_path: Optional[str] = None) -> None:
    """
    Create all tables if they don't already exist.
    Loads schema.sql from the same directory.
    
    Args:
        db_path: Optional path to database
    """
    if db_path is None:
        db_path = get_db_path()
    
    schema_path = Path(__file__).parent / "schema.sql"
    
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_path}. "
            "Please ensure schema.sql exists in data/sql/"
        )
    
    conn = connect(db_path)
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
        print(f"Database initialized at: {db_path}")
    finally:
        conn.close()


# Alias for backward compatibility
init_db = init_database


# ============================================================
# GENERIC QUERY WRAPPER
# ============================================================

def query(
    sql: str,
    params: Optional[tuple] = None,
    conn: Optional[sqlite3.Connection] = None
) -> pd.DataFrame:
    """
    Run a SQL query and return a DataFrame.
    
    Args:
        sql: SQL query string
        params: Optional tuple of query parameters
        conn: Optional existing connection
        
    Returns:
        pandas DataFrame with query results
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        if params:
            df = pd.read_sql_query(sql, conn, params=params)
        else:
            df = pd.read_sql_query(sql, conn)
        return df
    finally:
        if close_after:
            conn.close()


# ============================================================
# INDICATOR API
# ============================================================

def add_indicator(
    name: str,
    system: str,
    frequency: str = "daily",
    source: Optional[str] = None,
    units: Optional[str] = None,
    description: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Register a new indicator in the `indicators` table.
    
    Args:
        name: Indicator name (e.g., 'SPY', 'DGS10')
        system: System domain (e.g., 'market', 'economic')
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        source: Data source (e.g., 'yahoo', 'fred', 'stooq')
        units: Unit of measurement
        description: Human-readable description
        conn: Optional existing connection
        
    Returns:
        Row ID of the inserted/existing indicator
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO indicators (name, system, frequency, source, units, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, system, frequency, source, units, description),
        )
        conn.commit()
        
        # Get the ID (either from insert or existing)
        cursor = conn.execute(
            "SELECT id FROM indicators WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        return row["id"] if row else cursor.lastrowid
    finally:
        if close_after:
            conn.close()


def get_indicator(
    name: str,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[Dict[str, Any]]:
    """
    Get indicator metadata by name.
    
    Args:
        name: Indicator name
        conn: Optional existing connection
        
    Returns:
        Dictionary with indicator metadata or None if not found
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        cursor = conn.execute(
            "SELECT * FROM indicators WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        if close_after:
            conn.close()


def list_indicators(
    system: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    List all indicators, optionally filtered by system.
    
    Args:
        system: Optional system filter
        conn: Optional existing connection
        
    Returns:
        List of indicator metadata dictionaries
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        if system:
            cursor = conn.execute(
                "SELECT * FROM indicators WHERE system = ? ORDER BY name",
                (system,)
            )
        else:
            cursor = conn.execute("SELECT * FROM indicators ORDER BY system, name")
        
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if close_after:
            conn.close()


# ============================================================
# DATA WRITE/READ API
# ============================================================

def write_dataframe(
    df: pd.DataFrame,
    indicator: str,
    system: str,
    conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Write full time-series values for an indicator.
    
    The DataFrame must contain at minimum: [date, value]
    Optional columns: value_2, adjusted_value
    
    Args:
        df: DataFrame with date and value columns
        indicator: Indicator name
        system: System domain
        conn: Optional existing connection
        
    Returns:
        Number of rows written
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        # Validate required columns
        if "date" not in df.columns:
            raise ValueError("DataFrame must contain 'date' column")
        if "value" not in df.columns:
            raise ValueError("DataFrame must contain 'value' column")
        
        # Prepare records
        rows_written = 0
        for _, row in df.iterrows():
            date_val = str(row["date"])[:10]  # Ensure YYYY-MM-DD format
            value = float(row["value"]) if pd.notna(row["value"]) else None
            
            if value is None:
                continue  # Skip null values
            
            value_2 = float(row["value_2"]) if "value_2" in row and pd.notna(row.get("value_2")) else None
            adjusted = float(row["adjusted_value"]) if "adjusted_value" in row and pd.notna(row.get("adjusted_value")) else None
            
            conn.execute(
                """
                INSERT OR REPLACE INTO indicator_values 
                    (indicator, system, date, value, value_2, adjusted_value)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (indicator, system, date_val, value, value_2, adjusted),
            )
            rows_written += 1
        
        conn.commit()
        return rows_written
    finally:
        if close_after:
            conn.close()


def load_indicator(
    indicator: str,
    system: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> pd.DataFrame:
    """
    Load full time series for an indicator.
    
    Args:
        indicator: Indicator name
        system: Optional system filter
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        conn: Optional existing connection
        
    Returns:
        DataFrame with date and value columns
    """
    sql = """
        SELECT date, value, value_2, adjusted_value
        FROM indicator_values
        WHERE indicator = ?
    """
    params = [indicator]
    
    if system:
        sql += " AND system = ?"
        params.append(system)
    
    if start_date:
        sql += " AND date >= ?"
        params.append(start_date)
    
    if end_date:
        sql += " AND date <= ?"
        params.append(end_date)
    
    sql += " ORDER BY date ASC"
    
    df = query(sql, params=tuple(params), conn=conn)
    
    # Convert date to datetime
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    return df


def load_multiple_indicators(
    indicators: List[str],
    system: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> pd.DataFrame:
    """
    Load multiple indicators and pivot into a wide DataFrame.
    
    Args:
        indicators: List of indicator names
        system: Optional system filter
        start_date: Optional start date
        end_date: Optional end date
        conn: Optional existing connection
        
    Returns:
        DataFrame with date index and indicator columns
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        dfs = []
        for indicator in indicators:
            df = load_indicator(indicator, system, start_date, end_date, conn)
            if not df.empty:
                df = df[["date", "value"]].rename(columns={"value": indicator})
                df = df.set_index("date")
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine all indicators
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.join(df, how="outer")
        
        return combined.sort_index()
    finally:
        if close_after:
            conn.close()


# ============================================================
# FETCH LOGGING
# ============================================================

def log_fetch(
    indicator: str,
    system: str,
    source: str,
    rows_fetched: int,
    status: str = "success",
    error_message: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> None:
    """
    Log a fetch operation for auditing.
    
    Args:
        indicator: Indicator name
        system: System domain
        source: Data source used
        rows_fetched: Number of rows retrieved
        status: 'success', 'error', or 'partial'
        error_message: Optional error details
        conn: Optional existing connection
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        conn.execute(
            """
            INSERT INTO fetch_log 
                (indicator, system, source, fetch_date, rows_fetched, status, error_message)
            VALUES (?, ?, ?, date('now'), ?, ?, ?)
            """,
            (indicator, system, source, rows_fetched, status, error_message),
        )
        conn.commit()
    finally:
        if close_after:
            conn.close()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_date_range(
    indicator: str,
    system: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> tuple:
    """
    Get the date range for an indicator.
    
    Returns:
        Tuple of (min_date, max_date) or (None, None) if no data
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        sql = "SELECT MIN(date), MAX(date) FROM indicator_values WHERE indicator = ?"
        params = [indicator]
        
        if system:
            sql += " AND system = ?"
            params.append(system)
        
        cursor = conn.execute(sql, params)
        row = cursor.fetchone()
        return (row[0], row[1]) if row else (None, None)
    finally:
        if close_after:
            conn.close()


def export_to_csv(
    table_name: str,
    output_path: str,
    conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Export any table to CSV for debugging.
    
    Args:
        table_name: Name of table to export
        output_path: Path for output CSV file
        conn: Optional existing connection
        
    Returns:
        Number of rows exported
    """
    df = query(f"SELECT * FROM {table_name}", conn=conn)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} rows -> {output_path}")
    return len(df)


def get_table_stats(conn: Optional[sqlite3.Connection] = None) -> Dict[str, int]:
    """
    Get row counts for all tables.
    
    Returns:
        Dictionary mapping table names to row counts
    """
    close_after = False
    if conn is None:
        conn = connect()
        close_after = True
    
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        stats = {}
        for table in tables:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        
        return stats
    finally:
        if close_after:
            conn.close()


# ============================================================
# MAIN (For Manual Testing)
# ============================================================

if __name__ == "__main__":
    print("PRISM Database Module")
    print("=" * 50)
    print(f"Database path: {get_db_path()}\n")
    
    try:
        conn = connect()
        print("Connection: OK")
        
        # Show table stats
        stats = get_table_stats(conn)
        print("\nTable statistics:")
        for table, count in stats.items():
            print(f"  {table}: {count} rows")
        
        conn.close()
    except Exception as e:
        print(f"Connection: FAILED - {e}")
    
    print("\nQuick usage example:")
    print("  from data.sql.db import init_database, add_indicator, write_dataframe, load_indicator")
    print("  init_database()")
    print("  add_indicator('SPY', system='market', source='yahoo')")
    print("  write_dataframe(df, indicator='SPY', system='market')")
    print("  data = load_indicator('SPY')")
