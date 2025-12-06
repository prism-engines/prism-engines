"""
Unified Database API for PRISM Engine.

This module provides:
    - Modern unified accessors for indicator management
    - Fetch logging
    - Convenience helpers (quick_write)
    - Database statistics

IMPORTANT: This module imports ONLY from db_connector.py.
It does NOT import from prism_db.py to avoid circular dependencies.

For legacy compatibility, use prism_db.py directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

# -----------------------------------------------------------------------------
# IMPORT EVERYTHING FROM db_connector (MODERN API)
# This is the ONLY import source for db.py
# -----------------------------------------------------------------------------
from .db_connector import (
    # Connection
    get_connection,
    connect,
    # Initialization
    init_database,
    # Indicator registry
    add_indicator,
    get_indicator,
    list_indicators,
    # Fetch logging
    log_fetch,
    # Statistics
    database_stats,
    get_table_stats,
    get_date_range,
    # Data IO
    write_dataframe,
    load_indicator,
    load_multiple_indicators,
    query,
    export_to_csv,
)

from .db_path import get_db_path

_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# BACKWARD COMPATIBILITY ALIASES
# -----------------------------------------------------------------------------
init_db = init_database
initialize_db = init_database


# =============================================================================
# FETCH HISTORY QUERY
# =============================================================================
def get_fetch_history(
    entity: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Retrieve structured fetch logs.

    Args:
        entity: Filter by entity name
        source: Filter by data source
        limit: Maximum number of records to return

    Returns:
        DataFrame with fetch log entries
    """
    conn = get_connection()

    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_log'"
    ).fetchone()

    if exists is None:
        conn.close()
        return pd.DataFrame()

    sql = "SELECT * FROM fetch_log WHERE 1=1"
    params: List[Any] = []

    if entity:
        sql += " AND entity = ?"
        params.append(entity)

    if source:
        sql += " AND source = ?"
        params.append(source)

    sql += " ORDER BY completed_at DESC LIMIT ?"
    params.append(limit)

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# =============================================================================
# CONVENIENCE WRITER
# =============================================================================
def quick_write(
    df: pd.DataFrame,
    indicator_name: str,
    system: str,
    source: str,
    *,
    frequency: str = "daily",
    date_column: str = "date",
    value_column: str = "value",
) -> int:
    """
    Automatic indicator create + write + fetch_log.

    Args:
        df: DataFrame containing the data
        indicator_name: Name for the indicator
        system: System category ('market', 'econ', etc.)
        source: Data source name
        frequency: Data frequency (default 'daily')
        date_column: Name of date column in df
        value_column: Name of value column in df

    Returns:
        Number of rows written
    """
    started_at = datetime.now().isoformat()

    # Determine target table based on system
    if system == "market":
        table = "market_prices"
    elif system == "econ":
        table = "econ_values"
    else:
        table = "market_prices"  # Default

    try:
        # Ensure indicator is registered
        add_indicator(indicator_name, system=system, metadata={"source": source, "frequency": frequency})

        # Write the data
        rows = write_dataframe(df, table)

        log_fetch(
            source=source.lower(),
            entity=indicator_name,
            operation="fetch",
            status="success",
            rows_fetched=len(df),
            rows_inserted=rows,
            started_at=started_at,
        )

        return rows

    except Exception as e:
        log_fetch(
            source=source.lower(),
            entity=indicator_name,
            operation="fetch",
            status="error",
            error_message=str(e),
            started_at=started_at,
        )
        raise


# =============================================================================
# MODULE EXPORT LIST
# =============================================================================
__all__ = [
    # Core connection
    "get_connection",
    "connect",
    "get_db_path",

    # Initialization
    "init_database",
    "init_db",
    "initialize_db",

    # Indicator operations
    "add_indicator",
    "get_indicator",
    "list_indicators",

    # Data IO
    "write_dataframe",
    "load_indicator",
    "load_multiple_indicators",
    "query",
    "export_to_csv",

    # Fetch logging
    "log_fetch",
    "get_fetch_history",

    # Statistics
    "database_stats",
    "get_table_stats",
    "get_date_range",

    # Convenience
    "quick_write",
]
