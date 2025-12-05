"""
Geometry Engine Inputs for PRISM Engine.

Provides a unified adapter for fetching input series for the geometry engine.
All data access goes through this module - no ad-hoc queries in the geometry engine.

Usage:
    from engine_core.geometry.inputs import get_geometry_input_series

    inputs = get_geometry_input_series(conn, reg)
    # inputs["t10y2y"] -> DataFrame(date, value)
    # inputs["sector_breadth"] -> DataFrame(date, value)
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Categories of inputs useful for geometry analysis
GEOMETRY_INPUT_CATEGORIES = {
    "spreads": [
        "t10y2y",
        "t10y3m",
        "credit_spread",
        "real_yield_10y",
        "yield_curve_slope",
    ],
    "inflation": [
        "breakeven_inflation",
        "cpi",
        "ppi",
    ],
    "employment": [
        "employment_spread",
        "unrate",
        "payems",
    ],
    "liquidity": [
        "liquidity_ratio",
        "m2",
    ],
    "sector_technicals": [
        "sector_breadth",
        "sector_mom_diff",
    ],
    "equity_technicals": [
        "spy_mom_12m",
        "spy_mom_6m",
        "spy_vol_20d",
        "spy_vol_60d",
        "spy_rsi_14",
        "spy_roc_1m",
        "qqq_mom_12m",
        "qqq_vol_20d",
    ],
    "market_prices": [
        "spy",
        "qqq",
        "iwm",
        "vix",
        "gld",
        "tlt",
        "dxy",
    ],
    "rates": [
        "dgs10",
        "dgs2",
        "dgs3mo",
    ],
}


def load_series_from_db(
    conn: sqlite3.Connection,
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a single time series from the database.

    Args:
        conn: SQLite connection.
        name: Indicator name.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        DataFrame with 'date' and 'value' columns. Empty if not found.
    """
    query = """
        SELECT iv.date, iv.value
        FROM indicator_values iv
        JOIN indicators i ON iv.indicator_id = i.id
        WHERE i.name = ?
    """
    params = [name]

    if start_date:
        query += " AND iv.date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND iv.date <= ?"
        params.append(end_date)

    query += " ORDER BY iv.date"

    try:
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.warning(f"Error loading '{name}': {e}")
        return pd.DataFrame()


def get_geometry_input_series(
    conn: sqlite3.Connection,
    reg: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    categories: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Get all input series for the geometry engine.

    Returns a dictionary mapping metric name to DataFrame(date, value).
    The geometry engine reads from this adapter, not ad-hoc queries.

    Args:
        conn: SQLite database connection.
        reg: Metric registry dictionary.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        categories: Optional list of categories to include. If None, includes all.

    Returns:
        Dictionary mapping metric name to DataFrame with date and value columns.
    """
    results: dict[str, pd.DataFrame] = {}

    # Determine which categories to load
    if categories is None:
        categories = list(GEOMETRY_INPUT_CATEGORIES.keys())

    # Collect all metric names to load
    metrics_to_load: set[str] = set()
    for category in categories:
        if category in GEOMETRY_INPUT_CATEGORIES:
            metrics_to_load.update(GEOMETRY_INPUT_CATEGORIES[category])

    # Also include any synthetic and technical metrics from registry
    for section in ["synthetic", "technical"]:
        for metric in reg.get(section, []):
            name = metric.get("name")
            if name:
                metrics_to_load.add(name)

    logger.info(f"Loading {len(metrics_to_load)} geometry input series")

    # Load each metric
    for name in sorted(metrics_to_load):
        df = load_series_from_db(conn, name, start_date, end_date)
        if not df.empty:
            results[name] = df
            logger.debug(f"Loaded '{name}': {len(df)} rows")
        else:
            logger.debug(f"No data for '{name}'")

    logger.info(f"Loaded {len(results)} series with data")
    return results


def get_aligned_geometry_inputs(
    conn: sqlite3.Connection,
    reg: dict,
    metrics: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    align_method: str = "inner",
) -> pd.DataFrame:
    """
    Get multiple metrics aligned by date into a single DataFrame.

    Args:
        conn: SQLite connection.
        reg: Metric registry.
        metrics: List of metric names to include.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        align_method: How to align dates ('inner', 'outer', 'left', 'right').

    Returns:
        DataFrame with date index and one column per metric.
    """
    dfs = []

    for name in metrics:
        df = load_series_from_db(conn, name, start_date, end_date)
        if df.empty:
            logger.warning(f"No data for metric '{name}'")
            continue

        df = df.set_index("date").rename(columns={"value": name})
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Join all dataframes
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how=align_method)

    return result.sort_index()


def get_geometry_input_summary(
    conn: sqlite3.Connection,
    reg: dict,
) -> pd.DataFrame:
    """
    Get summary statistics for all geometry input series.

    Args:
        conn: SQLite connection.
        reg: Metric registry.

    Returns:
        DataFrame with metric name, row count, date range, mean, std.
    """
    inputs = get_geometry_input_series(conn, reg)

    summaries = []
    for name, df in inputs.items():
        if df.empty:
            continue

        summary = {
            "metric": name,
            "rows": len(df),
            "start_date": df["date"].min().strftime("%Y-%m-%d"),
            "end_date": df["date"].max().strftime("%Y-%m-%d"),
            "mean": df["value"].mean(),
            "std": df["value"].std(),
            "min": df["value"].min(),
            "max": df["value"].max(),
        }
        summaries.append(summary)

    return pd.DataFrame(summaries)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data.registry import load_metric_registry, validate_registry
    from data.sql.prism_db import get_db_path, init_db

    print("Geometry Input Series Loader")
    print("=" * 50)

    try:
        registry = load_metric_registry()
        validate_registry(registry)

        db_path = get_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Load all inputs
        inputs = get_geometry_input_series(conn, registry)

        print(f"\nLoaded {len(inputs)} input series:")
        print("-" * 50)

        for category, metrics in GEOMETRY_INPUT_CATEGORIES.items():
            available = [m for m in metrics if m in inputs]
            print(f"\n{category}:")
            for m in available:
                df = inputs[m]
                print(f"  {m:25s}: {len(df):6d} rows")

        # Get summary
        print("\n" + "=" * 50)
        print("Summary Statistics:")
        summary = get_geometry_input_summary(conn, registry)
        if not summary.empty:
            print(summary.to_string(index=False))

        conn.close()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
