"""
Sector Technicals for PRISM Engine.

Computes sector-level technical indicators driven by the registry.
- sector_mom_diff: Aggregated sector momentum differential vs SPY
- sector_breadth: Percentage of sectors above 200d moving average

Usage:
    from engine_core.metrics.sector_technicals import (
        compute_sector_momentum_diff,
        compute_sector_breadth,
    )
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Sector ETFs in the registry
SECTOR_ETFS = [
    "xly", "xlp", "xlf", "xlk", "xle",
    "xlb", "xli", "xlv", "xlc", "xlre", "xlu"
]


def load_indicator_series(
    conn: sqlite3.Connection,
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load indicator as DataFrame with date and value."""
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
        df = df.set_index("date")
    return df


def write_indicator_series(
    conn: sqlite3.Connection,
    name: str,
    data: pd.DataFrame,
    system: str = "finance",
    frequency: str = "daily",
    source: str = "computed",
) -> int:
    """Write computed indicator to database."""
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
            (name, system, frequency, source, f"Technical: {name}"),
        )
        indicator_id = cursor.lastrowid

    # Write values
    rows_written = 0
    for date, row_data in data.iterrows():
        value = row_data["value"] if "value" in row_data else row_data.iloc[0]
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


def compute_momentum(prices: pd.Series, period_months: int) -> pd.Series:
    """
    Compute momentum as percentage change over period.

    Args:
        prices: Price series.
        period_months: Lookback period in months (approx 21 trading days/month).

    Returns:
        Momentum series (percentage change).
    """
    days = period_months * 21
    return prices.pct_change(periods=days) * 100


def compute_rolling_volatility(prices: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling volatility (annualized standard deviation of returns).

    Args:
        prices: Price series.
        window: Rolling window in days.

    Returns:
        Annualized volatility series.
    """
    returns = prices.pct_change()
    return returns.rolling(window=window).std() * np.sqrt(252) * 100


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.

    Args:
        prices: Price series.
        period: RSI period (default 14).

    Returns:
        RSI series (0-100).
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_rate_of_change(prices: pd.Series, period_months: int) -> pd.Series:
    """
    Compute rate of change over period.

    Args:
        prices: Price series.
        period_months: Lookback period in months.

    Returns:
        ROC series (percentage).
    """
    days = period_months * 21
    return (prices / prices.shift(days) - 1) * 100


def compute_sector_momentum_diff(
    conn: sqlite3.Connection,
    reg: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute sector momentum differential vs SPY.

    For each sector ETF, computes 6-12 month momentum relative to SPY,
    then aggregates into a single indicator.

    Args:
        conn: SQLite connection.
        reg: Metric registry.
        start_date: Optional start date.
        end_date: Optional end date.

    Returns:
        DataFrame with date index and sector_mom_diff column.
    """
    # Get sectors from registry
    technical_metrics = reg.get("technical", [])
    sector_deps = []
    for metric in technical_metrics:
        if metric.get("name") == "sector_mom_diff":
            sector_deps = metric.get("depends_on", [])
            break

    if not sector_deps:
        sector_deps = SECTOR_ETFS

    # Load SPY as benchmark
    spy_df = load_indicator_series(conn, "spy", start_date, end_date)
    if spy_df.empty:
        logger.warning("No SPY data for sector momentum diff")
        return pd.DataFrame()

    # Load all sector ETFs
    sector_dfs = {}
    for sector in sector_deps:
        df = load_indicator_series(conn, sector, start_date, end_date)
        if not df.empty:
            sector_dfs[sector] = df

    if not sector_dfs:
        logger.warning("No sector data found")
        return pd.DataFrame()

    # Compute 6-month and 12-month momentum for SPY
    spy_mom_6m = compute_momentum(spy_df["value"], 6)
    spy_mom_12m = compute_momentum(spy_df["value"], 12)
    spy_avg_mom = (spy_mom_6m + spy_mom_12m) / 2

    # Compute relative momentum for each sector
    relative_moms = []
    for sector, df in sector_dfs.items():
        # Align dates with SPY
        aligned = df.join(spy_df, lsuffix="_sector", rsuffix="_spy", how="inner")
        if aligned.empty:
            continue

        sector_mom_6m = compute_momentum(aligned["value_sector"], 6)
        sector_mom_12m = compute_momentum(aligned["value_sector"], 12)
        sector_avg_mom = (sector_mom_6m + sector_mom_12m) / 2

        # Relative momentum vs SPY
        spy_aligned = spy_avg_mom.reindex(sector_avg_mom.index)
        relative = sector_avg_mom - spy_aligned
        relative_moms.append(relative.rename(sector))

    if not relative_moms:
        return pd.DataFrame()

    # Aggregate: mean of all sector relative momentums
    combined = pd.concat(relative_moms, axis=1)
    result = pd.DataFrame({"value": combined.mean(axis=1)})
    result = result.dropna()

    return result


def compute_sector_breadth(
    conn: sqlite3.Connection,
    reg: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute sector breadth: percentage of sectors above 200-day MA.

    Args:
        conn: SQLite connection.
        reg: Metric registry.
        start_date: Optional start date.
        end_date: Optional end date.

    Returns:
        DataFrame with date index and breadth value (0-1).
    """
    # Get sectors from registry
    technical_metrics = reg.get("technical", [])
    sector_deps = []
    for metric in technical_metrics:
        if metric.get("name") == "sector_breadth":
            sector_deps = metric.get("depends_on", [])
            break

    if not sector_deps:
        sector_deps = SECTOR_ETFS

    # Load all sector ETFs
    sector_dfs = {}
    for sector in sector_deps:
        df = load_indicator_series(conn, sector, start_date, end_date)
        if not df.empty:
            sector_dfs[sector] = df

    if not sector_dfs:
        logger.warning("No sector data found for breadth")
        return pd.DataFrame()

    # For each sector, compute whether price is above 200d MA
    above_ma = []
    for sector, df in sector_dfs.items():
        ma_200 = df["value"].rolling(window=200).mean()
        is_above = (df["value"] > ma_200).astype(float)
        above_ma.append(is_above.rename(sector))

    if not above_ma:
        return pd.DataFrame()

    # Combine and compute breadth (percentage above)
    combined = pd.concat(above_ma, axis=1)
    breadth = combined.mean(axis=1)  # 0-1 range

    result = pd.DataFrame({"value": breadth})
    result = result.dropna()

    return result


def build_technical_indicators(
    conn: sqlite3.Connection,
    reg: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[str, int]:
    """
    Build all technical indicators defined in the registry.

    Args:
        conn: SQLite connection.
        reg: Metric registry.
        start_date: Optional start date.
        end_date: Optional end date.

    Returns:
        Dictionary mapping indicator name to rows written.
    """
    results: dict[str, int] = {}
    technical_metrics = reg.get("technical", [])

    for metric in technical_metrics:
        name = metric.get("name")
        depends_on = metric.get("depends_on", [])

        if not name or not depends_on:
            continue

        logger.info(f"Building technical: {name}")

        try:
            if name == "sector_mom_diff":
                df = compute_sector_momentum_diff(conn, reg, start_date, end_date)
            elif name == "sector_breadth":
                df = compute_sector_breadth(conn, reg, start_date, end_date)
            elif name.endswith("_mom_12m"):
                # 12-month momentum
                base = depends_on[0]
                base_df = load_indicator_series(conn, base, start_date, end_date)
                if not base_df.empty:
                    df = pd.DataFrame({"value": compute_momentum(base_df["value"], 12)})
                else:
                    df = pd.DataFrame()
            elif name.endswith("_mom_6m"):
                # 6-month momentum
                base = depends_on[0]
                base_df = load_indicator_series(conn, base, start_date, end_date)
                if not base_df.empty:
                    df = pd.DataFrame({"value": compute_momentum(base_df["value"], 6)})
                else:
                    df = pd.DataFrame()
            elif name.endswith("_vol_20d"):
                # 20-day volatility
                base = depends_on[0]
                base_df = load_indicator_series(conn, base, start_date, end_date)
                if not base_df.empty:
                    df = pd.DataFrame({"value": compute_rolling_volatility(base_df["value"], 20)})
                else:
                    df = pd.DataFrame()
            elif name.endswith("_vol_60d"):
                # 60-day volatility
                base = depends_on[0]
                base_df = load_indicator_series(conn, base, start_date, end_date)
                if not base_df.empty:
                    df = pd.DataFrame({"value": compute_rolling_volatility(base_df["value"], 60)})
                else:
                    df = pd.DataFrame()
            elif name.endswith("_rsi_14"):
                # 14-day RSI
                base = depends_on[0]
                base_df = load_indicator_series(conn, base, start_date, end_date)
                if not base_df.empty:
                    df = pd.DataFrame({"value": compute_rsi(base_df["value"], 14)})
                else:
                    df = pd.DataFrame()
            elif name.endswith("_roc_1m"):
                # 1-month rate of change
                base = depends_on[0]
                base_df = load_indicator_series(conn, base, start_date, end_date)
                if not base_df.empty:
                    df = pd.DataFrame({"value": compute_rate_of_change(base_df["value"], 1)})
                else:
                    df = pd.DataFrame()
            else:
                logger.warning(f"Unknown technical indicator: {name}")
                df = pd.DataFrame()

            if df.empty:
                results[name] = 0
                continue

            df = df.dropna()
            rows = write_indicator_series(conn, name, df)
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

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from data.registry import load_metric_registry, validate_registry
    from data.sql.prism_db import get_db_path, init_db

    print("Sector Technicals Builder")
    print("=" * 50)

    try:
        registry = load_metric_registry()
        validate_registry(registry)

        db_path = get_db_path()
        init_db(db_path)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        results = build_technical_indicators(registry, conn)

        print()
        print("Results:")
        print("-" * 50)
        for name, rows in results.items():
            status = "OK" if rows > 0 else "EMPTY"
            print(f"  {name:25s}: {rows:6d} rows [{status}]")

        conn.close()

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
