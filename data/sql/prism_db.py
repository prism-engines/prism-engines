"""

SQLite-based storage for indicators and time series data.

Key concept:
    `system` = top-level domain for an indicator (e.g., finance, climate, chemistry).
    A single database can store multiple systems side by side.

Usage:
    from data.sql.prism_db import add_indicator, write_dataframe, load_indicator

    # Add an indicator
    add_indicator(
        name="SPY",
        system="finance",
        frequency="daily",
        source="Yahoo Finance",
        units="USD",
        description="S&P 500 ETF"
    )

    # Write data
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "value": [100.0, 101.5]})
    write_dataframe(df, indicator_name="SPY", system="finance")

    # Load data
    data = load_indicator("SPY")
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# valid system (domain) types
system_types = [
    "finance",
    "climate",
    "chemistry",
    "anthropology",
    "biology",
    "physics",
]

# default database path (can be overridden with PRISM_DB environment variable)
_default_db_path = Path(__file__).parent / "prism.db"


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_path() -> Path:
    """
    Get the database path from environment variable or use default.

    The PRISM_DB environment variable can be used to specify a custom path.
    """
    env_path = os.environ.get("PRISM_DB")
    if env_path:
        return Path(env_path)
    return _default_db_path


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """
    Context manager for database connections.

    Args:
        db_path: Optional path to database. If None, uses get_db_path().

    Yields:
        sqlite3.Connection with row factory set to sqlite3.Row
    """
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize the database with the schema.

    Creates tables if they don't exist. Safe to call multiple times.

    Args:
        db_path: Optional path to database. If None, uses get_db_path().
    """
    schema_path = Path(__file__).parent / "prism_schema.sql"

    with get_connection(db_path) as conn:
        with open(schema_path) as f:
            conn.executescript(f.read())
        conn.commit()


# =============================================================================
# DEPRECATION HELPER
# =============================================================================

def _handle_panel_deprecation(
    system: Optional[str],
    panel: Optional[str],
    stacklevel: int = 3,
) -> str:
    """
    Handle the deprecated `panel` argument.

    Args:
        system: The new `system` argument value
        panel: The deprecated `panel` argument value
        stacklevel: Stack level for the warning

    Returns:
        The resolved system value

    Raises:
        ValueError: If neither system nor panel is provided
    """
    if panel is not None:
        warnings.warn(
            "'panel' is deprecated; use 'system' instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        if system is None:
            system = panel

    if system is None:
        raise ValueError("`system` is required (e.g., 'finance', 'climate').")

    if system not in system_types:
        raise ValueError(
            f"Unknown system: {system}. Valid values: {system_types}"
        )

    return system


# =============================================================================
# INDICATOR MANAGEMENT
# =============================================================================

def add_indicator(
    name: str,
    system: Optional[str] = None,
    *,
    panel: Optional[str] = None,  # Deprecated
    frequency: str = "daily",
    source: Optional[str] = None,
    units: Optional[str] = None,
    description: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Add a new indicator to the database.

    Args:
        name: Unique identifier for the indicator (e.g., "SPY", "GDP")
        system: Domain/system type (finance, climate, chemistry, etc.)
        panel: DEPRECATED - Use `system` instead
        frequency: Data frequency (daily, weekly, monthly, quarterly, yearly)
        source: Data source (e.g., "FRED", "Yahoo Finance")
        units: Unit of measurement (e.g., "USD", "percent")
        description: Human-readable description
        db_path: Optional path to database

    Returns:
        The ID of the newly created indicator

    Raises:
        ValueError: If system is invalid or name already exists
        sqlite3.IntegrityError: If indicator name already exists
    """
    system = _handle_panel_deprecation(system, panel)

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO indicators (name, system, frequency, source, units, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, system, frequency, source, units, description),
        )
        conn.commit()
        return cursor.lastrowid


def get_indicator(
    name: str,
    db_path: Optional[Path] = None,
) -> Optional[dict]:
    """
    Get indicator metadata by name.

    Args:
        name: Indicator name
        db_path: Optional path to database

    Returns:
        Dictionary with indicator metadata, or None if not found
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, name, system, frequency, source, units, description,
                   created_at, updated_at
            FROM indicators
            WHERE name = ?
            """,
            (name,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)


def list_indicators(
    system: Optional[str] = None,
    *,
    panel: Optional[str] = None,  # Deprecated
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    List all indicators, optionally filtered by system.

    Args:
        system: Filter by system/domain type
        panel: DEPRECATED - Use `system` instead
        db_path: Optional path to database

    Returns:
        DataFrame with indicator metadata
    """
    if panel is not None:
        warnings.warn(
            "'panel' is deprecated; use 'system' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if system is None:
            system = panel

    with get_connection(db_path) as conn:
        if system is not None:
            if system not in system_types:
                raise ValueError(
                    f"Unknown system: {system}. Valid values: {system_types}"
                )
            query = """
                SELECT id, name, system, frequency, source, units, description,
                       created_at, updated_at
                FROM indicators
                WHERE system = ?
                ORDER BY name
            """
            df = pd.read_sql_query(query, conn, params=(system,))
        else:
            query = """
                SELECT id, name, system, frequency, source, units, description,
                       created_at, updated_at
                FROM indicators
                ORDER BY system, name
            """
            df = pd.read_sql_query(query, conn)

        return df


def update_indicator(
    name: str,
    *,
    system: Optional[str] = None,
    panel: Optional[str] = None,  # Deprecated
    frequency: Optional[str] = None,
    source: Optional[str] = None,
    units: Optional[str] = None,
    description: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Update indicator metadata.

    Args:
        name: Indicator name to update
        system: New system/domain type
        panel: DEPRECATED - Use `system` instead
        frequency: New frequency
        source: New source
        units: New units
        description: New description
        db_path: Optional path to database

    Returns:
        True if indicator was updated, False if not found
    """
    if panel is not None:
        warnings.warn(
            "'panel' is deprecated; use 'system' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if system is None:
            system = panel

    if system is not None and system not in system_types:
        raise ValueError(
            f"Unknown system: {system}. Valid values: {system_types}"
        )

    updates = []
    params = []

    if system is not None:
        updates.append("system = ?")
        params.append(system)
    if frequency is not None:
        updates.append("frequency = ?")
        params.append(frequency)
    if source is not None:
        updates.append("source = ?")
        params.append(source)
    if units is not None:
        updates.append("units = ?")
        params.append(units)
    if description is not None:
        updates.append("description = ?")
        params.append(description)

    if not updates:
        return False

    params.append(name)

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            f"UPDATE indicators SET {', '.join(updates)} WHERE name = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_indicator(
    name: str,
    db_path: Optional[Path] = None,
) -> bool:
    """
    Delete an indicator and all its values.

    Args:
        name: Indicator name to delete
        db_path: Optional path to database

    Returns:
        True if indicator was deleted, False if not found
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM indicators WHERE name = ?",
            (name,),
        )
        conn.commit()
        return cursor.rowcount > 0


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def write_dataframe(
    df: pd.DataFrame,
    indicator_name: str,
    system: Optional[str] = None,
    *,
    panel: Optional[str] = None,  # Deprecated
    frequency: str = "daily",
    source: Optional[str] = None,
    units: Optional[str] = None,
    description: Optional[str] = None,
    date_column: str = "date",
    value_column: str = "value",
    create_if_missing: bool = True,
    db_path: Optional[Path] = None,
) -> int:
    """
    Write a DataFrame to the database.

    Args:
        df: DataFrame with date and value columns
        indicator_name: Name of the indicator
        system: System/domain type (required if creating new indicator)
        panel: DEPRECATED - Use `system` instead
        frequency: Data frequency
        source: Data source
        units: Unit of measurement
        description: Description
        date_column: Name of the date column in df
        value_column: Name of the value column in df
        create_if_missing: If True, create indicator if it doesn't exist
        db_path: Optional path to database

    Returns:
        Number of rows written

    Raises:
        ValueError: If indicator doesn't exist and system not provided
    """
    system = _handle_panel_deprecation(system, panel) if (system or panel) else None

    # Ensure database is initialized
    init_db(db_path)

    with get_connection(db_path) as conn:
        # Get or create indicator
        indicator = get_indicator(indicator_name, db_path)

        if indicator is None:
            if not create_if_missing:
                raise ValueError(f"Indicator '{indicator_name}' not found")

            if system is None:
                raise ValueError(
                    f"Indicator '{indicator_name}' not found and `system` "
                    "is required to create it."
                )

            indicator_id = add_indicator(
                name=indicator_name,
                system=system,
                frequency=frequency,
                source=source,
                units=units,
                description=description,
                db_path=db_path,
            )
        else:
            indicator_id = indicator["id"]

        # Prepare data
        data = df[[date_column, value_column]].copy()
        data.columns = ["date", "value"]
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
        data["indicator_id"] = indicator_id

        # Use INSERT OR REPLACE to handle duplicates
        rows_written = 0
        for _, row in data.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
                VALUES (?, ?, ?)
                """,
                (row["indicator_id"], row["date"], row["value"]),
            )
            rows_written += 1

        conn.commit()
        return rows_written


def load_indicator(
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load indicator data as a DataFrame.

    Args:
        name: Indicator name
        start_date: Optional start date (inclusive), ISO format
        end_date: Optional end date (inclusive), ISO format
        db_path: Optional path to database

    Returns:
        DataFrame with date index and value column

    Raises:
        ValueError: If indicator not found
    """
    with get_connection(db_path) as conn:
        # Get indicator ID
        indicator = get_indicator(name, db_path)
        if indicator is None:
            raise ValueError(f"Indicator '{name}' not found")

        # Build query
        query = """
            SELECT date, value
            FROM indicator_values
            WHERE indicator_id = ?
        """
        params = [indicator["id"]]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        return df


def load_multiple_indicators(
    names: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load multiple indicators and combine into a single DataFrame.

    Args:
        names: List of indicator names
        start_date: Optional start date (inclusive)
        end_date: Optional end date (inclusive)
        db_path: Optional path to database

    Returns:
        DataFrame with date index and one column per indicator
    """
    dfs = []
    for name in names:
        try:
            df = load_indicator(name, start_date, end_date, db_path)
            df = df.rename(columns={"value": name})
            dfs.append(df)
        except ValueError:
            warnings.warn(f"Indicator '{name}' not found, skipping")

    if not dfs:
        return pd.DataFrame()

    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how="outer")

    return result.sort_index()


def load_system_indicators(
    system: str,
    *,
    panel: Optional[str] = None,  # Deprecated
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load all indicators for a given system as a single DataFrame.

    Args:
        system: System/domain type (finance, climate, etc.)
        panel: DEPRECATED - Use `system` instead
        start_date: Optional start date (inclusive)
        end_date: Optional end date (inclusive)
        db_path: Optional path to database

    Returns:
        DataFrame with date index and one column per indicator
    """
    if panel is not None:
        warnings.warn(
            "'panel' is deprecated; use 'system' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if system is None:
            system = panel

    if system not in system_types:
        raise ValueError(
            f"Unknown system: {system}. Valid values: {system_types}"
        )

    indicators_df = list_indicators(system=system, db_path=db_path)
    if indicators_df.empty:
        return pd.DataFrame()

    return load_multiple_indicators(
        indicators_df["name"].tolist(),
        start_date,
        end_date,
        db_path,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_migration(
    migration_path: Path,
    db_path: Optional[Path] = None,
) -> None:
    """
    Run a migration SQL script.

    Args:
        migration_path: Path to the migration SQL file
        db_path: Optional path to database
    """
    with get_connection(db_path) as conn:
        with open(migration_path) as f:
            conn.executescript(f.read())
        conn.commit()


def get_db_stats(db_path: Optional[Path] = None) -> dict:
    """
    Get database statistics.

    Args:
        db_path: Optional path to database

    Returns:
        Dictionary with database statistics
    """
    with get_connection(db_path) as conn:
        indicator_count = conn.execute(
            "SELECT COUNT(*) FROM indicators"
        ).fetchone()[0]

        value_count = conn.execute(
            "SELECT COUNT(*) FROM indicator_values"
        ).fetchone()[0]

        system_counts = {}
        for row in conn.execute(
            "SELECT system, COUNT(*) as count FROM indicators GROUP BY system"
        ):
            system_counts[row["system"]] = row["count"]

        return {
            "indicator_count": indicator_count,
            "value_count": value_count,
            "systems": system_counts,
        }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Quick usage example
    print("PRISM Database Module")
    print("=" * 50)
    print()
    print("Available system types:", system_types)
    print()
    print("Usage:")
    print("  from data.sql.prism_db import add_indicator, write_dataframe, load_indicator")
    print()
    print("  add_indicator('SPY', system='finance', frequency='daily')")
    print("  write_dataframe(df, 'SPY', system='finance')")
    print("  data = load_indicator('SPY')")
PRISM Engine - SQL Database Manager
This file handles ALL database operations for PRISM.
Drop it in your project and import what you need.

SETUP (one time):
    1. Set your database location:
       export PRISM_DB="/mnt/chromeos/GoogleDrive/MyDrive/prismsql/prism.db"
    
    2. Initialize the database:
       from prism_db import init_database
       init_database()

DAILY USE:
    from prism_db import (
        connect,
        add_indicator,
        write_values,
        load_indicator,
        query,
        top_indicators,
        indicator_history
    )

Author: Jason Rudder / PRISM Engine
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_db_path():
    """
    Find the database file. Checks in order:
    1. PRISM_DB environment variable (recommended)
    2. Default location in home directory
    """
    env_path = os.environ.get('PRISM_DB')
    if env_path:
        return env_path
    
    home = Path.home()
    return str(home / 'prismsql' / 'prism.db')


def connect(db_path=None):
    """
    Connect to the database.
    
    Usage:
        conn = connect()
        # ... do stuff ...
        conn.close()
    """
    if db_path is None:
        db_path = get_db_path()
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path=None):
    """
    Create all tables if they don't exist.
    Safe to run multiple times - won't delete existing data.
    """
    if db_path is None:
        db_path = get_db_path()
    
    schema_path = Path(__file__).parent / 'prism_schema.sql'
    
    if not schema_path.exists():
        schema_path = Path('prism_schema.sql')
    
    if not schema_path.exists():
        print("ERROR: Can't find prism_schema.sql")
        print("Make sure it's in the same folder as prism_db.py")
        return False
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    conn = connect(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
        print(f"Database initialized at: {db_path}")
        return True
    except Exception as e:
        print(f"ERROR initializing database: {e}")
        return False
    finally:
        conn.close()


# =============================================================================
# WRITING DATA
# =============================================================================

def add_indicator(name, panel, frequency='daily', units=None, source=None, 
                  description=None, conn=None):
    """
    Add a new indicator to the database.
    If it already exists, returns the existing ID.
    
    Args:
        name: Indicator name (e.g., 'SPY', 'DXY', 'WALCL')
        panel: Category (e.g., 'equity', 'rates', 'liquidity', 'currency')
        frequency: 'daily', 'weekly', 'monthly', or 'quarterly'
        units: Optional - 'USD', 'percent', 'index', etc.
        source: Optional - 'FRED', 'Yahoo', etc.
        description: Optional - human-readable description
    
    Returns:
        The indicator's ID
    """
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        cursor.execute("""
            INSERT INTO indicators (name, panel, frequency, units, source, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, panel, frequency, units, source, description))
        
        conn.commit()
        indicator_id = cursor.lastrowid
        print(f"Added indicator: {name} (ID: {indicator_id})")
        return indicator_id
        
    finally:
        if close_conn:
            conn.close()


def write_values(indicator_name, df, date_column='date', value_column='value',
                 value_2_column=None, adjusted_column=None, conn=None):
    """
    Write time series data for an indicator.
    
    Args:
        indicator_name: Name of the indicator (must exist in database)
        df: Pandas DataFrame with the data
        date_column: Name of the date column (default: 'date')
        value_column: Name of the value column (default: 'value')
    """
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        row = cursor.fetchone()
        
        if not row:
            print(f"ERROR: Indicator '{indicator_name}' not found. Add it first with add_indicator()")
            return 0
        
        indicator_id = row[0]
        
        records = []
        for _, row_data in df.iterrows():
            date_val = row_data[date_column]
            if isinstance(date_val, pd.Timestamp):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)[:10]
            
            value = row_data[value_column] if pd.notna(row_data[value_column]) else None
            value_2 = row_data[value_2_column] if value_2_column and pd.notna(row_data.get(value_2_column)) else None
            adjusted = row_data[adjusted_column] if adjusted_column and pd.notna(row_data.get(adjusted_column)) else None
            
            records.append((indicator_id, date_str, value, value_2, adjusted))
        
        cursor.executemany("""
            INSERT INTO indicator_values (indicator_id, date, value, value_2, adjusted_value)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(indicator_id, date) DO UPDATE SET
                value = excluded.value,
                value_2 = excluded.value_2,
                adjusted_value = excluded.adjusted_value
        """, records)
        
        conn.commit()
        print(f"Wrote {len(records)} values for {indicator_name}")
        return len(records)
        
    finally:
        if close_conn:
            conn.close()


def write_dataframe(df, indicator_name, panel, frequency='daily', **kwargs):
    """
    Convenience function: Creates indicator if needed, then writes data.
    One-liner for importing data.
    
    Usage:
        write_dataframe(spy_df, 'SPY', 'equity', source='Yahoo')
    """
    conn = connect()
    try:
        add_indicator(indicator_name, panel, frequency, conn=conn, **kwargs)
        return write_values(indicator_name, df, conn=conn)
    finally:
        conn.close()


# =============================================================================
# READING DATA
# =============================================================================

def load_indicator(indicator_name, start_date=None, end_date=None, conn=None):
    """
    Load time series data for an indicator as a pandas DataFrame.
    
    Usage:
        df = load_indicator('SPY')
        df = load_indicator('SPY', start_date='2020-01-01')
    """
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        query_sql = """
            SELECT iv.date, iv.value, iv.value_2, iv.adjusted_value
            FROM indicator_values iv
            JOIN indicators i ON iv.indicator_id = i.id
            WHERE i.name = ?
        """
        params = [indicator_name]
        
        if start_date:
            query_sql += " AND iv.date >= ?"
            params.append(str(start_date)[:10])
        
        if end_date:
            query_sql += " AND iv.date <= ?"
            params.append(str(end_date)[:10])
        
        query_sql += " ORDER BY iv.date"
        
        df = pd.read_sql_query(query_sql, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        return df
        
    finally:
        if close_conn:
            conn.close()


def load_multiple(indicator_names, start_date=None, end_date=None, 
                  pivot=True, conn=None):
    """
    Load multiple indicators at once.
    
    Usage:
        df = load_multiple(['SPY', 'DXY', 'AGG'], start_date='2020-01-01')
    """
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        placeholders = ','.join(['?' for _ in indicator_names])
        query_sql = f"""
            SELECT i.name AS indicator, iv.date, iv.value
            FROM indicator_values iv
            JOIN indicators i ON iv.indicator_id = i.id
            WHERE i.name IN ({placeholders})
        """
        params = list(indicator_names)
        
        if start_date:
            query_sql += " AND iv.date >= ?"
            params.append(str(start_date)[:10])
        
        if end_date:
            query_sql += " AND iv.date <= ?"
            params.append(str(end_date)[:10])
        
        query_sql += " ORDER BY iv.date, i.name"
        
        df = pd.read_sql_query(query_sql, conn, params=params)
        df['date'] = pd.to_datetime(df['date'])
        
        if pivot:
            df = df.pivot(index='date', columns='indicator', values='value')
            df = df.reset_index()
        
        return df
        
    finally:
        if close_conn:
            conn.close()


def query(sql, params=None, conn=None):
    """
    Run any SQL query and return results as a DataFrame.
    
    Usage:
        df = query("SELECT * FROM indicators")
        df = query("SELECT * FROM indicators WHERE panel = ?", ['equity'])
    """
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        if params:
            return pd.read_sql_query(sql, conn, params=params)
        else:
            return pd.read_sql_query(sql, conn)
    finally:
        if close_conn:
            conn.close()


def list_indicators(panel=None, conn=None):
    """List all indicators in the database."""
    if panel:
        return query("SELECT * FROM indicators WHERE panel = ? ORDER BY name", [panel], conn)
    else:
        return query("SELECT * FROM indicators ORDER BY panel, name", conn=conn)


# =============================================================================
# ENGINE OUTPUT HELPERS
# =============================================================================

def save_window(start_date, end_date, label=None, n_observations=None, conn=None):
    """Create or get a temporal analysis window. Returns window_id."""
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        if label is None:
            label = f"{start_date[:4]}-{end_date[:4]}"
        
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        
        cursor.execute("""
            INSERT OR IGNORE INTO windows (start_date, end_date, start_year, end_year, label, n_observations)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (start_date, end_date, start_year, end_year, label, n_observations))
        
        cursor.execute("SELECT id FROM windows WHERE start_date = ? AND end_date = ?", 
                      (start_date, end_date))
        window_id = cursor.fetchone()[0]
        
        conn.commit()
        return window_id
        
    finally:
        if close_conn:
            conn.close()


def save_consensus(window_id, results_df, conn=None):
    """
    Save consensus rankings for a window.
    
    Args:
        window_id: ID of the window
        results_df: DataFrame with columns: indicator, avg_rank, std_rank, etc.
    """
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        for _, row in results_df.iterrows():
            indicator_name = row.get('indicator', row.get('name'))
            
            cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
            ind_row = cursor.fetchone()
            if not ind_row:
                continue
            indicator_id = ind_row[0]
            
            cursor.execute("""
                INSERT OR REPLACE INTO consensus 
                (window_id, indicator_id, avg_rank, median_rank, std_rank, 
                 min_rank, max_rank, agreement_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                window_id, indicator_id,
                row.get('avg_rank'), row.get('median_rank'), row.get('std_rank'),
                row.get('min_rank'), row.get('max_rank'), row.get('agreement_score')
            ))
        
        conn.commit()
        print(f"Saved consensus for {len(results_df)} indicators")
        
    finally:
        if close_conn:
            conn.close()


def save_lens_result(window_id, lens_name, indicator_name, rank, raw_score=None, 
                     normalized_score=None, conn=None):
    """Save a single lens result."""
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM lenses WHERE name = ?", (lens_name,))
        lens_row = cursor.fetchone()
        if not lens_row:
            print(f"ERROR: Lens '{lens_name}' not found")
            return
        lens_id = lens_row[0]
        
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,))
        ind_row = cursor.fetchone()
        if not ind_row:
            print(f"ERROR: Indicator '{indicator_name}' not found")
            return
        indicator_id = ind_row[0]
        
        cursor.execute("""
            INSERT OR REPLACE INTO lens_results 
            (window_id, indicator_id, lens_id, rank, raw_score, normalized_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (window_id, indicator_id, lens_id, rank, raw_score, normalized_score))
        
        conn.commit()
        
    finally:
        if close_conn:
            conn.close()


def save_regime_transition(window_before_id, window_after_id, spearman_corr, 
                           top_10_overlap=None, regime_break=False, notes=None, conn=None):
    """Save regime stability metrics between two windows."""
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT end_year FROM windows WHERE id = ?", (window_before_id,))
        transition_year = cursor.fetchone()[0]
        
        cursor.execute("""
            INSERT OR REPLACE INTO regime_stability 
            (window_before_id, window_after_id, transition_year, spearman_corr, 
             top_10_overlap, regime_break_flag, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (window_before_id, window_after_id, transition_year, spearman_corr,
              top_10_overlap, 1 if regime_break else 0, notes))
        
        conn.commit()
        
    finally:
        if close_conn:
            conn.close()


# =============================================================================
# QUERY HELPERS
# =============================================================================

def top_indicators(window_id=None, n=10, conn=None):
    """
    Get top-ranked indicators.
    
    Usage:
        df = top_indicators()  # Top 10 from most recent window
        df = top_indicators(window_id=5, n=20)
    """
    if window_id is None:
        sql = """
            SELECT i.name AS indicator, i.panel, c.avg_rank, c.std_rank, 
                   c.agreement_score, w.label AS window
            FROM consensus c
            JOIN indicators i ON c.indicator_id = i.id
            JOIN windows w ON c.window_id = w.id
            WHERE c.window_id = (SELECT id FROM windows ORDER BY end_date DESC LIMIT 1)
            ORDER BY c.avg_rank ASC
            LIMIT ?
        """
        return query(sql, [n], conn)
    else:
        sql = """
            SELECT i.name AS indicator, i.panel, c.avg_rank, c.std_rank, 
                   c.agreement_score, w.label AS window
            FROM consensus c
            JOIN indicators i ON c.indicator_id = i.id
            JOIN windows w ON c.window_id = w.id
            WHERE c.window_id = ?
            ORDER BY c.avg_rank ASC
            LIMIT ?
        """
        return query(sql, [window_id, n], conn)


def indicator_history(indicator_name, conn=None):
    """Get ranking history for a single indicator across all windows."""
    sql = """
        SELECT w.start_year, w.end_year, w.label, c.avg_rank, c.std_rank
        FROM consensus c
        JOIN indicators i ON c.indicator_id = i.id
        JOIN windows w ON c.window_id = w.id
        WHERE i.name = ?
        ORDER BY w.start_year
    """
    return query(sql, [indicator_name], conn)


def regime_breaks(threshold=0.3, conn=None):
    """Find regime breaks (low correlation between consecutive windows)."""
    sql = """
        SELECT transition_year, spearman_corr, top_10_overlap, notes
        FROM regime_stability
        WHERE spearman_corr < ?
        ORDER BY transition_year
    """
    return query(sql, [threshold], conn)


def coherence_events(min_score=None, event_type=None, conn=None):
    """Get coherence events (when lenses agreed)."""
    sql = "SELECT * FROM coherence_events WHERE 1=1"
    params = []
    
    if min_score:
        sql += " AND coherence_score >= ?"
        params.append(min_score)
    
    if event_type:
        sql += " AND event_type = ?"
        params.append(event_type)
    
    sql += " ORDER BY date DESC"
    
    return query(sql, params if params else None, conn)


# =============================================================================
# UTILITIES
# =============================================================================

def database_stats(conn=None):
    """Get overview statistics of the database."""
    close_conn = False
    if conn is None:
        conn = connect()
        close_conn = True
    
    try:
        stats = {}
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM indicators")
        stats['n_indicators'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM indicator_values")
        stats['n_data_points'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM windows")
        stats['n_windows'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date), MAX(date) FROM indicator_values")
        row = cursor.fetchone()
        stats['date_range'] = (row[0], row[1])
        
        cursor.execute("SELECT DISTINCT panel FROM indicators")
        stats['panels'] = [r[0] for r in cursor.fetchall()]
        
        return stats
        
    finally:
        if close_conn:
            conn.close()


def export_to_csv(table_name, output_path, conn=None):
    """Export any table to CSV."""
    df = query(f"SELECT * FROM {table_name}", conn=conn)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} rows to {output_path}")


# =============================================================================
# MAIN - Run this file directly to test
# =============================================================================

if __name__ == '__main__':
    print("PRISM Database Manager")
    print("=" * 50)
    print(f"Database path: {get_db_path()}")
    print()
    
    try:
        conn = connect()
        print("Connection: OK")
        conn.close()
    except Exception as e:
        print(f"Connection: FAILED - {e}")
    
    print()
    print("Quick Start:")
    print("  from prism_db import init_database, add_indicator, write_values, load_indicator")
    print("  init_database()  # Run once to create tables")
