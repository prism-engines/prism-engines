"""
Geometry Engine Inputs
======================

Prepares and validates input series for the geometry engine.

The geometry engine performs multi-dimensional analysis of indicator
relationships. This module ensures all required input series are
available and properly formatted.

Schema v2 Notes:
- Uses indicator_values.indicator_name as the key (not indicator_id)
- Joins to indicators table via indicators.name
- No numeric IDs used for linking tables
"""

from __future__ import annotations

import logging
import sqlite3
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_geometry_input_series(
    conn: sqlite3.Connection,
    registry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Get all available input series for the geometry engine.

    Verifies that required indicators exist in the database and have
    sufficient data for geometric analysis.

    Args:
        conn: SQLite database connection
        registry: The metric registry

    Returns:
        List of dictionaries containing series metadata:
            - name: Indicator name
            - system: System/domain type
            - count: Number of data points
            - start_date: First available date
            - end_date: Last available date
    """
    cursor = conn.cursor()

    # Check if indicators table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='indicators'"
    )
    if not cursor.fetchone():
        logger.info("No indicators table found - geometry inputs not yet available")
        return []

    # Check if indicator_values table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='indicator_values'"
    )
    if not cursor.fetchone():
        logger.info("No indicator_values table found - geometry inputs not yet available")
        return []

    # Get all indicators from database (Schema v2 - uses indicator_name, not indicator_id)
    cursor.execute(
        """
        SELECT
            iv.indicator_name,
            i.system,
            COUNT(*) as count,
            MIN(iv.date) as start_date,
            MAX(iv.date) as end_date
        FROM indicator_values iv
        LEFT JOIN indicators i ON iv.indicator_name = i.name
        GROUP BY iv.indicator_name
        HAVING count > 0
        ORDER BY i.system, iv.indicator_name
        """
    )

    rows = cursor.fetchall()
    inputs = []

    for row in rows:
        inputs.append({
            "name": row[0],
            "system": row[1] or "unknown",  # May be NULL if not in indicators table
            "count": row[2],
            "start_date": row[3],
            "end_date": row[4],
        })

    logger.info(f"Found {len(inputs)} indicators available for geometry engine")

    return inputs


def validate_geometry_inputs(
    inputs: List[Dict[str, Any]],
    min_observations: int = 252,
) -> List[str]:
    """
    Validate that input series meet minimum requirements.

    Args:
        inputs: List of input series metadata
        min_observations: Minimum number of observations required

    Returns:
        List of indicator names that meet requirements
    """
    valid = []

    for inp in inputs:
        if inp.get("count", 0) >= min_observations:
            valid.append(inp["name"])
        else:
            logger.debug(
                f"Indicator {inp['name']} has only {inp.get('count', 0)} "
                f"observations (need {min_observations})"
            )

    logger.info(f"{len(valid)} indicators meet minimum observation threshold")

    return valid


def load_geometry_matrix(
    conn: sqlite3.Connection,
    indicator_names: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a matrix of indicator values for geometric analysis.

    Schema v2: Uses indicator_values.indicator_name directly (no join needed for values).

    Args:
        conn: SQLite database connection
        indicator_names: List of indicator names to include
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with date index and one column per indicator (columns = indicator names)
    """
    if not indicator_names:
        return pd.DataFrame()

    placeholders = ",".join(["?" for _ in indicator_names])

    # Schema v2: Query indicator_values directly using indicator_name
    query = f"""
        SELECT iv.indicator_name, iv.date, iv.value
        FROM indicator_values iv
        WHERE iv.indicator_name IN ({placeholders})
    """
    params = list(indicator_names)

    if start_date:
        query += " AND iv.date >= ?"
        params.append(start_date)

    if end_date:
        query += " AND iv.date <= ?"
        params.append(end_date)

    query += " ORDER BY iv.indicator_name, iv.date"

    df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    # Pivot to wide format (columns = indicator_name)
    df["date"] = pd.to_datetime(df["date"])
    matrix = df.pivot(index="date", columns="indicator_name", values="value")

    return matrix
