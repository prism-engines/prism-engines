"""
Synthetic Time Series Pipeline
==============================

Builds synthetic (derived) time series from base indicators.

Synthetic indicators are computed from existing data rather than fetched
from external sources. Examples include:
    - Spreads (e.g., 10Y-2Y yield spread)
    - Ratios (e.g., equity/bond ratio)
    - Transforms (e.g., YoY percent change)
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def build_synthetic_timeseries(
    registry: Dict[str, Any],
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, int]:
    """
    Build synthetic time series from base indicators.

    Args:
        registry: The metric registry containing synthetic definitions
        conn: SQLite database connection
        start_date: Start date for computation (YYYY-MM-DD)
        end_date: End date for computation (YYYY-MM-DD)

    Returns:
        Dictionary mapping synthetic indicator names to row counts written
    """
    results: Dict[str, int] = {}
    synthetic_defs = registry.get("synthetic", [])

    if not synthetic_defs:
        logger.info("No synthetic indicators defined in registry")
        return results

    cursor = conn.cursor()

    for synth in synthetic_defs:
        name = synth.get("name")
        formula = synth.get("formula", "")
        inputs = synth.get("inputs", [])

        if not name:
            continue

        try:
            # Load input series
            input_data = {}
            for input_name in inputs:
                cursor.execute(
                    """
                    SELECT iv.date, iv.value
                    FROM indicator_values iv
                    JOIN indicators i ON iv.indicator_id = i.id
                    WHERE i.name = ?
                    ORDER BY iv.date
                    """,
                    (input_name,),
                )
                rows = cursor.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=["date", "value"])
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                    input_data[input_name] = df["value"]

            if not input_data:
                logger.warning(f"No input data for synthetic {name}")
                results[name] = 0
                continue

            # Compute synthetic based on formula type
            output_series = _compute_synthetic(formula, input_data)

            if output_series is None or output_series.empty:
                logger.warning(f"Empty result for synthetic {name}")
                results[name] = 0
                continue

            # Get or create indicator
            cursor.execute("SELECT id FROM indicators WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                indicator_id = row[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO indicators (name, system, frequency, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, "finance", "daily", "synthetic"),
                )
                indicator_id = cursor.lastrowid

            # Write values
            count = 0
            for date_val, value in output_series.items():
                if pd.isna(value):
                    continue
                date_str = date_val.strftime("%Y-%m-%d")
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
                    VALUES (?, ?, ?)
                    """,
                    (indicator_id, date_str, float(value)),
                )
                count += 1

            conn.commit()
            results[name] = count
            logger.info(f"Built synthetic {name}: {count} rows")

        except Exception as e:
            logger.error(f"Error building synthetic {name}: {e}")
            results[name] = 0

    return results


def _compute_synthetic(formula: str, input_data: Dict[str, pd.Series]) -> Optional[pd.Series]:
    """
    Compute a synthetic series based on formula type.

    Supported formulas:
        - spread: difference between two series
        - ratio: ratio of two series
        - yoy: year-over-year percent change
        - mom: month-over-month percent change
        - log_return: log returns
    """
    formula = formula.lower()
    keys = list(input_data.keys())

    if not keys:
        return None

    # Align all series to common dates
    df = pd.DataFrame(input_data)

    if formula == "spread" and len(keys) >= 2:
        return df[keys[0]] - df[keys[1]]

    elif formula == "ratio" and len(keys) >= 2:
        return df[keys[0]] / df[keys[1]]

    elif formula == "yoy":
        return df[keys[0]].pct_change(periods=252) * 100

    elif formula == "mom":
        return df[keys[0]].pct_change(periods=21) * 100

    elif formula == "log_return":
        import numpy as np
        return np.log(df[keys[0]] / df[keys[0]].shift(1))

    else:
        # Default: return first series unchanged
        logger.warning(f"Unknown formula '{formula}', returning first input unchanged")
        return df[keys[0]]
