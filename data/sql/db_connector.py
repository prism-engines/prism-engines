"""
db_connector.py  
Modern indicator-management API for PRISM Engine.

This module provides:
  • indicator registry: add/get/list/update/delete
  • bulk loading utilities
  • DB statistics

It uses the underlying SQLite connection from prism_db.
"""

import sqlite3
from pathlib import Path
from typing import Optional, List
import pandas as pd

from .prism_db import get_connection, get_db_path


# =============================================================================
# INDICATOR REGISTRY
# =============================================================================

def add_indicator(name: str, system: str, source: str, frequency: str = "daily"):
    """
    Create an indicator entry in the indicators table.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO indicators (name, system, source, frequency)
            VALUES (?, ?, ?, ?)
            """,
            (name, system, source, frequency),
        )
        conn.commit()


def get_indicator(name: str) -> Optional[dict]:
    """
    Return indicator metadata.
    """

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM indicators WHERE name = ?", (name,)
        ).fetchone()

        if row is None:
            return None

        return dict(row)


def list_indicators(system: Optional[str] = None) -> List[str]:
    """
    List indicators (optionally filtered by system).
    """

    with get_connection() as conn:
        if system:
            rows = conn.execute(
                "SELECT name FROM indicators WHERE system = ?", (system,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT name FROM indicators"
            ).fetchall()

        return [r[0] for r in rows]


def update_indicator(name: str, **kwargs):
    """
    Update indicator metadata (system, source, frequency).
    """

    fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [name]

    with get_connection() as conn:
        conn.execute(
            f"UPDATE indicators SET {fields} WHERE name = ?",
            values,
        )
        conn.commit()


def delete_indicator(name: str):
    """
    Delete indicator metadata (does NOT delete data rows).
    """

    with get_connection() as conn:
        conn.execute("DELETE FROM indicators WHERE name = ?", (name,))
        conn.commit()


# =============================================================================
# BULK LOADERS
# =============================================================================

def load_multiple_indicators(names: list[str]) -> pd.DataFrame:
    """
    Load multiple indicators into a single DataFrame.
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

        ORDER BY date ASC
    """

    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=names + names)

    return df


def load_system_indicators(system: str) -> pd.DataFrame:
    """
    Load all indicators associated with a given system.
    """

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM indicators WHERE system = ?", (system,)
        ).fetchall()

    names = [r[0] for r in rows]
    return load_multiple_indicators(names)


# =============================================================================
# DB STATISTICS
# =============================================================================

def get_db_stats() -> pd.DataFrame:
    """
    Return row counts per table in the DB.
    """

    with get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        stats = []
        for t in tables:
            table = t[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats.append({"table": table, "rows": count})

        return pd.DataFrame(stats)
