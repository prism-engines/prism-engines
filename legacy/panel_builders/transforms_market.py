"""
PRISM Market Transforms
=======================

DEPRECATED: This module uses domain-specific logic (market) and will be removed.
The new architecture is domain-agnostic - all indicators are treated equally.

Transformation utilities for market data series.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Deprecation warning
warnings.warn(
    "panel.transforms_market is DEPRECATED. "
    "The new architecture is domain-agnostic - use generic transforms instead.",
    DeprecationWarning,
    stacklevel=2
)

logger = logging.getLogger(__name__)


def align_market_series(
    series_dict: dict[str, pd.DataFrame],
    date_column: str = "date",
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Align multiple market series to a common date index.

    Takes a dictionary of DataFrames (each with date and value columns)
    and outer-joins them on the date index.

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
            # Single column, use it as value
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
        f"Aligned {len(aligned_dfs)} market series, "
        f"resulting in {len(result)} rows"
    )

    return result


def compute_returns(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "simple",
    periods: int = 1,
) -> pd.DataFrame:
    """
    Compute returns for specified columns.

    Args:
        df: DataFrame with price data
        columns: Columns to compute returns for (default: all numeric)
        method: 'simple' for arithmetic returns, 'log' for log returns
        periods: Number of periods for return calculation

    Returns:
        DataFrame with return columns (suffixed with _ret)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        if method == "log":
            result[f"{col}_ret"] = np.log(df[col] / df[col].shift(periods))
        else:
            result[f"{col}_ret"] = df[col].pct_change(periods=periods)

    return result


def compute_moving_average(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    window: int = 20,
    ma_type: str = "simple",
) -> pd.DataFrame:
    """
    Compute moving averages for specified columns.

    Args:
        df: DataFrame with price data
        columns: Columns to compute MA for (default: all numeric)
        window: Rolling window size
        ma_type: 'simple' for SMA, 'exponential' for EMA

    Returns:
        DataFrame with MA columns (suffixed with _ma{window})
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        suffix = f"_ma{window}"
        if ma_type == "exponential":
            result[f"{col}{suffix}"] = df[col].ewm(span=window).mean()
        else:
            result[f"{col}{suffix}"] = df[col].rolling(window=window).mean()

    return result


def compute_volatility(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    window: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling volatility for specified columns.

    Args:
        df: DataFrame with return data
        columns: Columns to compute volatility for
        window: Rolling window size
        annualize: If True, annualize volatility (assumes 252 trading days)

    Returns:
        DataFrame with volatility columns (suffixed with _vol)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(index=df.index)
    annualization_factor = np.sqrt(252) if annualize else 1.0

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        result[f"{col}_vol"] = (
            df[col].rolling(window=window).std() * annualization_factor
        )

    return result


def normalize_series(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "zscore",
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Normalize series values.

    Args:
        df: DataFrame with data
        columns: Columns to normalize
        method: 'zscore' for standard normalization,
                'minmax' for 0-1 scaling,
                'rank' for percentile rank
        window: If provided, use rolling window for calculations

    Returns:
        DataFrame with normalized columns (suffixed with _norm)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found, skipping")
            continue

        series = df[col]

        if method == "zscore":
            if window:
                mean = series.rolling(window=window).mean()
                std = series.rolling(window=window).std()
                result[f"{col}_norm"] = (series - mean) / std
            else:
                result[f"{col}_norm"] = (series - series.mean()) / series.std()

        elif method == "minmax":
            if window:
                min_val = series.rolling(window=window).min()
                max_val = series.rolling(window=window).max()
                result[f"{col}_norm"] = (series - min_val) / (max_val - min_val)
            else:
                result[f"{col}_norm"] = (
                    (series - series.min()) / (series.max() - series.min())
                )

        elif method == "rank":
            if window:
                result[f"{col}_norm"] = series.rolling(window=window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                    raw=False,
                )
            else:
                result[f"{col}_norm"] = series.rank(pct=True)

    return result
