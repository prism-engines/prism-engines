"""
Synthetic Time Series Pipeline
==============================

Builds synthetic (derived) time series from base indicators.

Synthetic indicators are computed from existing data rather than fetched
from external sources. Examples include:
    - Spreads (e.g., 10Y-2Y yield spread)
    - Ratios (e.g., equity/bond ratio)
    - Transforms (e.g., YoY percent change)
    - Real yields (nominal - inflation)
    - Liquidity ratios
    - Employment spreads

Supports the Full Institutional Pack with 250+ indicators.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np
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

    Supports both legacy registry format and new YAML-based format
    with explicit formula and inputs fields.

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

    # Build set of available indicators for dependency checking
    available_indicators = _get_available_indicators(cursor)
    logger.debug(f"Found {len(available_indicators)} available indicators in database")

    # Sort synthetics by dependency order to handle cascading calculations
    sorted_synthetics = _topological_sort_synthetics(synthetic_defs, available_indicators)

    for synth in sorted_synthetics:
        name = synth.get("name")
        formula = synth.get("formula", "")
        # Support both 'inputs' (new) and 'depends_on' (legacy) field names
        inputs = synth.get("inputs") or synth.get("depends_on", [])

        if not name:
            continue

        try:
            # Load input series
            input_data = _load_input_series(cursor, inputs, start_date, end_date)

            if not input_data:
                logger.debug(f"No input data for synthetic {name} (inputs: {inputs})")
                results[name] = 0
                continue

            # Compute synthetic based on formula type
            output_series = _compute_synthetic(formula, input_data, synth)

            if output_series is None or output_series.empty:
                logger.debug(f"Empty result for synthetic {name}")
                results[name] = 0
                continue

            # Write results
            count = _write_synthetic_values(cursor, name, output_series)
            conn.commit()

            results[name] = count
            if count > 0:
                logger.info(f"Built synthetic {name}: {count} rows")
                # Add to available indicators for downstream synthetics
                available_indicators.add(name)

        except Exception as e:
            logger.error(f"Error building synthetic {name}: {e}")
            results[name] = 0

    return results


def _get_available_indicators(cursor: sqlite3.Cursor) -> set:
    """Get set of indicator names available in the database."""
    cursor.execute("SELECT name FROM indicators")
    return {row[0] for row in cursor.fetchall()}


def _topological_sort_synthetics(
    synthetics: List[Dict],
    available: set
) -> List[Dict]:
    """
    Sort synthetic indicators by dependency order.

    This ensures that if synthetic A depends on synthetic B,
    B is computed before A.
    """
    # Build dependency graph
    synth_by_name = {s.get("name"): s for s in synthetics if s.get("name")}
    synth_names = set(synth_by_name.keys())

    # Simple topological sort using Kahn's algorithm
    sorted_list = []
    ready = []
    pending = []

    for s in synthetics:
        name = s.get("name")
        if not name:
            continue

        inputs = s.get("inputs") or s.get("depends_on", [])
        # Check if all dependencies are either available or not synthetic
        deps_satisfied = all(
            inp in available or inp not in synth_names
            for inp in inputs
        )

        if deps_satisfied:
            ready.append(s)
        else:
            pending.append(s)

    # Process ready queue
    processed = set()
    while ready:
        current = ready.pop(0)
        name = current.get("name")
        sorted_list.append(current)
        processed.add(name)

        # Check if any pending items are now ready
        still_pending = []
        for s in pending:
            inputs = s.get("inputs") or s.get("depends_on", [])
            synth_deps = [inp for inp in inputs if inp in synth_names]

            if all(dep in processed or dep in available for dep in synth_deps):
                ready.append(s)
            else:
                still_pending.append(s)
        pending = still_pending

    # Add any remaining (may have missing deps, will fail gracefully)
    sorted_list.extend(pending)

    return sorted_list


def _load_input_series(
    cursor: sqlite3.Cursor,
    inputs: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.Series]:
    """Load input series from database (Schema v2 - uses indicator_name)."""
    input_data = {}

    for input_name in inputs:
        # Schema v2: Query indicator_values directly using indicator_name
        query = """
            SELECT iv.date, iv.value
            FROM indicator_values iv
            WHERE iv.indicator_name = ?
        """
        params = [input_name]

        if start_date:
            query += " AND iv.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND iv.date <= ?"
            params.append(end_date)

        query += " ORDER BY iv.date"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if rows:
            df = pd.DataFrame(rows, columns=["date", "value"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            input_data[input_name] = df["value"]

    return input_data


def _write_synthetic_values(
    cursor: sqlite3.Cursor,
    name: str,
    series: pd.Series
) -> int:
    """Write synthetic series values to database (Schema v2 - uses indicator_name)."""
    # Ensure indicator exists in registry
    cursor.execute("SELECT name FROM indicators WHERE name = ?", (name,))
    row = cursor.fetchone()

    if not row:
        cursor.execute(
            """
            INSERT INTO indicators (name, source)
            VALUES (?, ?)
            """,
            (name, "synthetic"),
        )

    # Schema v2: Write values using indicator_name directly
    count = 0
    for date_val, value in series.items():
        if pd.isna(value) or np.isinf(value):
            continue
        date_str = date_val.strftime("%Y-%m-%d")
        cursor.execute(
            """
            INSERT OR REPLACE INTO indicator_values (indicator_name, date, value, provenance)
            VALUES (?, ?, ?, ?)
            """,
            (name, date_str, float(value), "synthetic"),
        )
        count += 1

    return count


def _compute_synthetic(
    formula: str,
    input_data: Dict[str, pd.Series],
    synth_def: Dict[str, Any] = None
) -> Optional[pd.Series]:
    """
    Compute a synthetic series based on formula type.

    Supported formulas:
        - spread: difference between two series (A - B)
        - ratio: ratio of two series (A / B)
        - inverse_ratio: inverse ratio (B / A)
        - yoy: year-over-year percent change (252 trading days)
        - mom: month-over-month percent change (21 trading days)
        - qoq: quarter-over-quarter percent change (63 trading days)
        - log_return: log returns
        - zscore: rolling z-score normalization
        - diff: first difference
        - sum: sum of multiple series
        - product: product of multiple series
        - max: max of multiple series
        - min: min of multiple series
    """
    if not formula:
        formula = "spread"  # Default to spread for backward compatibility

    formula = formula.lower().strip()
    keys = list(input_data.keys())

    if not keys:
        return None

    # Align all series to common dates
    df = pd.DataFrame(input_data)

    # Get optional parameters from synth_def
    params = synth_def.get("params", {}) if synth_def else {}

    # Basic binary operations
    if formula == "spread" and len(keys) >= 2:
        return df[keys[0]] - df[keys[1]]

    elif formula == "ratio" and len(keys) >= 2:
        denominator = df[keys[1]].replace(0, np.nan)
        return df[keys[0]] / denominator

    elif formula == "inverse_ratio" and len(keys) >= 2:
        denominator = df[keys[0]].replace(0, np.nan)
        return df[keys[1]] / denominator

    # Percentage changes
    elif formula == "yoy":
        periods = params.get("periods", 252)
        return df[keys[0]].pct_change(periods=periods) * 100

    elif formula == "mom":
        periods = params.get("periods", 21)
        return df[keys[0]].pct_change(periods=periods) * 100

    elif formula == "qoq":
        periods = params.get("periods", 63)
        return df[keys[0]].pct_change(periods=periods) * 100

    elif formula == "log_return":
        series = df[keys[0]]
        return np.log(series / series.shift(1))

    elif formula == "cumulative_return":
        series = df[keys[0]]
        return (series / series.iloc[0] - 1) * 100

    # Statistical transforms
    elif formula == "zscore":
        window = params.get("window", 63)
        series = df[keys[0]]
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)

    elif formula == "percentile":
        window = params.get("window", 252)
        series = df[keys[0]]
        return series.rolling(window=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

    elif formula == "diff":
        periods = params.get("periods", 1)
        return df[keys[0]].diff(periods=periods)

    elif formula == "rolling_mean":
        window = params.get("window", 20)
        return df[keys[0]].rolling(window=window).mean()

    elif formula == "rolling_std":
        window = params.get("window", 20)
        return df[keys[0]].rolling(window=window).std()

    # Aggregation operations
    elif formula == "sum":
        return df[keys].sum(axis=1)

    elif formula == "product":
        return df[keys].prod(axis=1)

    elif formula == "max":
        return df[keys].max(axis=1)

    elif formula == "min":
        return df[keys].min(axis=1)

    elif formula == "mean":
        return df[keys].mean(axis=1)

    # Index operations
    elif formula == "rebase":
        # Rebase series to start at 100
        series = df[keys[0]]
        first_valid = series.first_valid_index()
        if first_valid is not None:
            return (series / series.loc[first_valid]) * 100
        return series

    # Default: return first series unchanged
    else:
        logger.warning(f"Unknown formula '{formula}', returning first input unchanged")
        return df[keys[0]]


def get_synthetic_summary(results: Dict[str, int]) -> Dict[str, Any]:
    """Generate summary statistics for synthetic build results."""
    successful = {k: v for k, v in results.items() if v > 0}
    failed = {k: v for k, v in results.items() if v == 0}

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_rows": sum(results.values()),
        "successful_indicators": list(successful.keys()),
        "failed_indicators": list(failed.keys()),
    }
