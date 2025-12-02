"""
PRISM Database Module
=====================

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
