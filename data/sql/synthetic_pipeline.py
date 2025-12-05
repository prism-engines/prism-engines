"""
Synthetic Pipeline for PRISM Engine.

Builds synthetic time series from registry-defined dependencies.
All computations are driven by the registry's synthetic section - no hardcoded lists.

Usage:
    from data.sql.synthetic_pipeline import build_synthetic_timeseries
    from data.registry import load_metric_registry

    reg = load_metric_registry()
    conn = sqlite3.connect("prism.db")
    build_synthetic_timeseries(reg, conn)
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional, Callable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# SYNTHETIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_t10y2y(deps: pd.DataFrame) -> pd.Series:
    """10-Year minus 2-Year Treasury spread."""
    return deps["dgs10"] - deps["dgs2"]


def compute_t10y3m(deps: pd.DataFrame) -> pd.Series:
    """10-Year minus 3-Month Treasury spread."""
    return deps["dgs10"] - deps["dgs3mo"]


def compute_credit_spread(deps: pd.DataFrame) -> pd.Series:
    """Corporate credit spread (BAA minus 10Y Treasury)."""
    return deps["baaffm"] - deps["dgs10"]


def compute_real_yield_10y(deps: pd.DataFrame) -> pd.Series:
    """Real yield = 10Y Treasury minus YoY CPI inflation."""
    # Calculate YoY CPI change (percentage)
    cpi_yoy = deps["cpi"].pct_change(periods=12) * 100
    return deps["dgs10"] - cpi_yoy


def compute_breakeven_inflation(deps: pd.DataFrame) -> pd.Series:
    """Breakeven inflation proxy from CPI and PPI."""
    # Simple proxy: average of CPI and PPI YoY changes
    cpi_yoy = deps["cpi"].pct_change(periods=12) * 100
    ppi_yoy = deps["ppi"].pct_change(periods=12) * 100
    return (cpi_yoy + ppi_yoy) / 2


def compute_employment_spread(deps: pd.DataFrame) -> pd.Series:
    """Standardized spread between unemployment rate and payrolls."""
    # Standardize both series
    unrate_z = (deps["unrate"] - deps["unrate"].mean()) / deps["unrate"].std()
    payems_z = (deps["payems"] - deps["payems"].mean()) / deps["payems"].std()
    # Employment spread: negative unrate (lower is better) plus payems
    return payems_z - unrate_z


def compute_liquidity_ratio(deps: pd.DataFrame) -> pd.Series:
    """M2 money supply relative to GDP."""
    # Forward-fill GDP to align with more frequent M2
    gdp_filled = deps["gdp"].ffill()
    # Ratio in trillions
    return deps["m2"] / gdp_filled


def compute_yield_curve_slope(deps: pd.DataFrame) -> pd.Series:
    """Yield curve slope (same as t10y2y but semantically distinct)."""
    return deps["dgs10"] - deps["dgs2"]


# Mapping of synthetic metric names to computation functions
SYNTHETIC_COMPUTATIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "t10y2y": compute_t10y2y,
    "t10y3m": compute_t10y3m,
    "credit_spread": compute_credit_spread,
    "real_yield_10y": compute_real_yield_10y,
    "breakeven_inflation": compute_breakeven_inflation,
    "employment_spread": compute_employment_spread,
    "liquidity_ratio": compute_liquidity_ratio,
    "yield_curve_slope": compute_yield_curve_slope,
}


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def load_indicator_data(
    conn: sqlite3.Connection,
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load indicator data from database.

    Args:
        conn: SQLite connection.
        name: Indicator name.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        DataFrame with date and value columns.
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

    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_dependencies(
    conn: sqlite3.Connection,
    dep_names: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load multiple dependencies and join on date.

    Args:
        conn: SQLite connection.
        dep_names: List of indicator names to load.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        DataFrame with date index and one column per dependency.
    """
    dfs = []

    for name in dep_names:
        df = load_indicator_data(conn, name, start_date, end_date)
        if df.empty:
            logger.warning(f"No data found for dependency: {name}")
            continue
        df = df.set_index("date").rename(columns={"value": name})
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Inner join on date to get aligned data
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how="inner")

    return result


def write_synthetic_series(
    conn: sqlite3.Connection,
    name: str,
    data: pd.DataFrame,
    system: str = "finance",
    frequency: str = "daily",
    source: str = "synthetic",
) -> int:
    """
    Write a synthetic series to the database.

    Args:
        conn: SQLite connection.
        name: Synthetic indicator name.
        data: DataFrame with date index and 'value' column.
        system: System category (default: 'finance').
        frequency: Data frequency (default: 'daily').
        source: Data source (default: 'synthetic').

    Returns:
        Number of rows written.
    """
    cursor = conn.cursor()

    # Get or create indicator
    cursor.execute("SELECT id FROM indicators WHERE name = ?", (name,))
    row = cursor.fetchone()

    if row:
        indicator_id = row[0]
    else:
        cursor.execute(
            """
            INSERT INTO indicators (name, system, frequency, source, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, system, frequency, source, f"Synthetic: {name}"),
        )
        indicator_id = cursor.lastrowid
        logger.info(f"Created indicator: {name} (id={indicator_id})")

    # Prepare and write data
    rows_written = 0
    for date, row_data in data.iterrows():
        value = row_data["value"]
        if pd.isna(value):
            continue

        date_str = date.strftime("%Y-%m-%d")
        cursor.execute(
            """
            INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
            VALUES (?, ?, ?)
            """,
            (indicator_id, date_str, float(value)),
        )
        rows_written += 1

    conn.commit()
    return rows_written


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_synthetic_timeseries(
    reg: dict,
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, int]:
    """
    Build all synthetic time series defined in the registry.

    Loops through the 'synthetic' section of the registry, loads dependencies,
    computes synthetic values, and writes to database.

    Args:
        reg: The metric registry dictionary.
        conn: SQLite database connection.
        start_date: Optional start date for data.
        end_date: Optional end date for data.

    Returns:
        Dictionary mapping synthetic name to rows written.
    """
    synthetic_metrics = reg.get("synthetic", [])
    results: dict[str, int] = {}

    logger.info(f"Building {len(synthetic_metrics)} synthetic series")

    for metric in synthetic_metrics:
        name = metric.get("name")
        depends_on = metric.get("depends_on", [])

        if not name:
            logger.warning("Synthetic metric missing 'name', skipping")
            continue

        if not depends_on:
            logger.warning(f"Synthetic '{name}' has no dependencies, skipping")
            continue

        # Check if we have a computation function
        compute_fn = SYNTHETIC_COMPUTATIONS.get(name)
        if compute_fn is None:
            logger.warning(f"No computation function for synthetic '{name}', skipping")
            continue

        logger.info(f"Building synthetic: {name} (deps: {depends_on})")

        # Load dependencies
        deps_df = load_dependencies(conn, depends_on, start_date, end_date)
        if deps_df.empty:
            logger.warning(f"No data for dependencies of '{name}'")
            results[name] = 0
            continue

        # Check all dependencies are present
        missing = set(depends_on) - set(deps_df.columns)
        if missing:
            logger.warning(f"Missing dependencies for '{name}': {missing}")
            results[name] = 0
            continue

        try:
            # Compute synthetic values
            values = compute_fn(deps_df)

            # Prepare output dataframe
            output = pd.DataFrame({"value": values}, index=deps_df.index)
            output = output.dropna()

            if output.empty:
                logger.warning(f"No valid values computed for '{name}'")
                results[name] = 0
                continue

            # Determine frequency from first dependency
            first_dep = reg.get("economic", [])
            freq = "daily"
            for item in first_dep:
                if item.get("name") == depends_on[0]:
                    freq = item.get("frequency", "daily")
                    break

            # Write to database
            rows = write_synthetic_series(conn, name, output, frequency=freq)
            results[name] = rows
            logger.info(f"  Wrote {rows} rows for '{name}'")

        except Exception as e:
            logger.error(f"Error computing '{name}': {e}")
            results[name] = 0

    return results


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

    # Import registry loader
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data.registry import load_metric_registry, validate_registry
    from data.sql.prism_db import get_db_path, init_db

    print("Synthetic Pipeline Builder")
    print("=" * 50)

    try:
        # Load and validate registry
        registry = load_metric_registry()
        validate_registry(registry)
        print("Registry loaded and validated")

        # Initialize database
        db_path = get_db_path()
        init_db(db_path)
        print(f"Database: {db_path}")

        # Build synthetics
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        results = build_synthetic_timeseries(registry, conn)

        print()
        print("Results:")
        print("-" * 50)
        total = 0
        for name, rows in results.items():
            status = "OK" if rows > 0 else "EMPTY"
            print(f"  {name:25s}: {rows:6d} rows [{status}]")
            total += rows

        print("-" * 50)
        print(f"  {'TOTAL':25s}: {total:6d} rows")

        conn.close()

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
