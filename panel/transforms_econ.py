"""
PRISM Economic Transforms
=========================

Transformation utilities for economic data series.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def align_econ_series(
    series_dict: dict[str, pd.DataFrame],
    date_column: str = "date",
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Align multiple economic series to a common date index.

    Similar to market alignment but handles different frequencies
    (monthly, weekly, quarterly) by outer-joining on dates.

    Args:
        series_dict: Dictionary mapping series names to DataFrames
        date_column: Name of the date column in each DataFrame
        value_column: Name of the value column in each DataFrame

    Returns:
        DataFrame with date index and one column per series
    """
    if not series_dict:
        logger.warning("No series provided for alignment")
        return pd.DataFrame()

    aligned_dfs = []

    for name, df in series_dict.items():
        if df is None or df.empty:
            logger.warning(f"Skipping empty series: {name}")
            continue

        # Normalize to date index
        series_df = df.copy()

        # Handle both indexed and column-based dates
        if date_column in series_df.columns:
            series_df[date_column] = pd.to_datetime(series_df[date_column])
            series_df = series_df.set_index(date_column)

        # Select value column and rename
        if value_column in series_df.columns:
            series_df = series_df[[value_column]].rename(
                columns={value_column: name}
            )
        elif len(series_df.columns) == 1:
            series_df.columns = [name]
        else:
            logger.warning(
                f"Cannot determine value column for {name}, skipping"
            )
            continue

        aligned_dfs.append(series_df)

    if not aligned_dfs:
        logger.warning("No valid series to align")
        return pd.DataFrame()

    # Outer join all series
    result = aligned_dfs[0]
    for df in aligned_dfs[1:]:
        result = result.join(df, how="outer")

    # Sort by date
    result = result.sort_index()

    logger.info(
        f"Aligned {len(aligned_dfs)} economic series, "
        f"resulting in {len(result)} rows"
    )

    return result


def forward_fill_to_daily(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Forward-fill lower frequency data onto a daily calendar.

    Useful for aligning monthly/weekly economic data with daily
    market data.

    Args:
        df: DataFrame with date index (can be sparse)
        columns: Columns to forward-fill (default: all)
        start_date: Start date for daily calendar
        end_date: End date for daily calendar
        limit: Maximum number of days to forward-fill

    Returns:
        DataFrame reindexed to daily frequency with forward-filled values
    """
    if df.empty:
        return df

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Determine date range
    if start_date is None:
        start_date = df.index.min()
    else:
        start_date = pd.to_datetime(start_date)

    if end_date is None:
        end_date = df.index.max()
    else:
        end_date = pd.to_datetime(end_date)

    # Create daily calendar
    daily_index = pd.date_range(start=start_date, end=end_date, freq="D")

    # Reindex to daily and forward-fill
    result = df.reindex(daily_index)

    if columns is not None:
        for col in columns:
            if col in result.columns:
                result[col] = result[col].ffill(limit=limit)
    else:
        result = result.ffill(limit=limit)

    logger.info(
        f"Forward-filled to daily frequency: {len(result)} rows "
        f"from {start_date.date()} to {end_date.date()}"
    )

    return result


def compute_yoy_change(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    periods: int = 12,
) -> pd.DataFrame:
    """
    Compute year-over-year percentage change.

    For monthly data, uses 12-period lag.
    For quarterly data, uses 4-period lag.

    Args:
        df: DataFrame with economic data
        columns: Columns to compute YoY for (default: all numeric)
        periods: Number of periods representing one year

    Returns:
        DataFrame with YoY columns (suffixed with _yoy)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        result[f"{col}_yoy"] = df[col].pct_change(periods=periods) * 100

    return result


def compute_mom_change(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute month-over-month change (absolute difference).

    Args:
        df: DataFrame with economic data
        columns: Columns to compute MoM for (default: all numeric)

    Returns:
        DataFrame with MoM columns (suffixed with _mom)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        result[f"{col}_mom"] = df[col].diff()

    return result


def compute_zscore(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    window: Optional[int] = None,
    expanding: bool = False,
) -> pd.DataFrame:
    """
    Compute z-score normalization.

    Args:
        df: DataFrame with economic data
        columns: Columns to normalize (default: all numeric)
        window: Rolling window for calculating mean/std
        expanding: If True and window is None, use expanding window

    Returns:
        DataFrame with z-score columns (suffixed with _z)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        series = df[col]

        if window:
            mean = series.rolling(window=window).mean()
            std = series.rolling(window=window).std()
        elif expanding:
            mean = series.expanding().mean()
            std = series.expanding().std()
        else:
            mean = series.mean()
            std = series.std()

        result[f"{col}_z"] = (series - mean) / std

    return result


def detect_frequency(series: pd.Series) -> str:
    """
    Detect the frequency of a time series.

    Args:
        series: Series with datetime index

    Returns:
        Frequency string: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    """
    if len(series) < 2:
        return "unknown"

    # Calculate median gap between observations
    gaps = series.index.to_series().diff().dropna()
    median_gap = gaps.median()

    if median_gap <= pd.Timedelta(days=1):
        return "daily"
    elif median_gap <= pd.Timedelta(days=7):
        return "weekly"
    elif median_gap <= pd.Timedelta(days=31):
        return "monthly"
    elif median_gap <= pd.Timedelta(days=92):
        return "quarterly"
    else:
        return "yearly"


def interpolate_series(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "linear",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Interpolate missing values in economic series.

    Args:
        df: DataFrame with economic data
        columns: Columns to interpolate (default: all numeric)
        method: Interpolation method ('linear', 'time', 'spline')
        limit: Maximum number of consecutive NaNs to fill

    Returns:
        DataFrame with interpolated values
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue

        if method == "spline":
            result[col] = result[col].interpolate(
                method="spline", order=3, limit=limit
            )
        else:
            result[col] = result[col].interpolate(method=method, limit=limit)

    return result
