"""
Unified Database API for PRISM Engine.

This module provides:
    - Modern unified accessors for indicator management
    - Legacy compatibility bridging for prism_db
    - Fetch logging
    - Convenience helpers (quick_write)

It intentionally does NOT rely on prism_db for indicator logic.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# -----------------------------------------------------------------------------
# IMPORT THE LEGACY MODULE (ONLY FOR CONNECTION + BASIC IO)
# -----------------------------------------------------------------------------
from .prism_db import (
    get_connection,
    get_db_path,
    init_db,
    write_dataframe,
    load_indicator,
    run_all_migrations as run_migration,
)

# -----------------------------------------------------------------------------
# INDICATOR MANAGEMENT — PROVIDED BY db_connector (MODERN API)
# -----------------------------------------------------------------------------
try:
    from .db_connector import (
        add_indicator,
        get_indicator,
        list_indicators,
        update_indicator,
        delete_indicator,
        load_multiple_indicators,
        load_system_indicators,
        get_db_stats,
    )
except ImportError:
    # Fail loudly so users know the dependency is missing
    raise ImportError(
        "db_connector.py is missing or incomplete. "
        "The unified SQL layer requires db_connector for indicator operations."
    )

# Backward compatibility alias
initialize_db = init_db

_logger = logging.getLogger(__name__)


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
    db_path: Optional[Path] = None,
) -> int:
    """
    Log a fetch operation in the fetch_log table.
    """

    if started_at is None:
        started_at = datetime.now().isoformat()

    with get_connection() as conn:
        # Ensure table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_log'"
        ).fetchone()

        if exists is None:
            _logger.warning(
                "fetch_log table not found — run migrations to enable fetch logging."
            )
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
        return cursor.lastrowid


# =============================================================================
# FETCH HISTORY QUERY
# =============================================================================
def get_fetch_history(
    entity: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Retrieve structured fetch logs.
    """
    with get_connection() as conn:

        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_log'"
        ).fetchone()

        if exists is None:
            return pd.DataFrame()

        query = "SELECT * FROM fetch_log WHERE 1=1"
        params = []

        if entity:
            query += " AND entity = ?"
            params.append(entity)

        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY completed_at DESC LIMIT ?"
        params.append(limit)

        return pd.read_sql_query(query, conn, params=params)


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
    db_path: Optional[Path] = None,
) -> int:
    """
    Automatic indicator create + write + fetch_log.
    """

    started_at = datetime.now().isoformat()

    try:
        rows = write_dataframe(
            df,
            indicator_name,
        )

        log_fetch(
            source=source.lower(),
            entity=indicator_name,
            operation="fetch",
            status="success",
            rows_fetched=len(df),
            rows_inserted=rows,
            started_at=started_at,
            db_path=db_path,
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
            db_path=db_path,
        )
        raise


# =============================================================================
# MODULE EXPORT LIST
# =============================================================================
__all__ = [
    # Core connection
    "get_connection",
    "get_db_path",
    "init_db",
    "initialize_db",

    # Indicator operations (from db_connector)
    "add_indicator",
    "get_indicator",
    "list_indicators",
    "update_indicator",
    "delete_indicator",
    "load_multiple_indicators",
    "load_system_indicators",
    "get_db_stats",

    # Data IO
    "write_dataframe",
    "load_indicator",

    # Fetch logging
    "log_fetch",
    "get_fetch_history",

    # Utils
    "run_migration",

    # Convenience
    "quick_write",
]
