"""
Total Return Index (TRI) Calculation - PRISM Approved Method

Uses additive decomposition: Price Return + Income Return = Total Return
Matches MSCI/S&P daily TRI methodology.
"""

from __future__ import annotations
import pandas as pd


def compute_total_return(
    df: pd.DataFrame,
    price_col: str = "close",
    yield_col: str = "yield",
    trading_days: int = 252,
    base_value: float = 100.0,
) -> pd.DataFrame:
    """
    Compute Total Return Index (TRI) from price and yield series.

    Args:
        df: DataFrame with price and yield columns
        price_col: Name of price column
        yield_col: Name of yield column (decimal form, e.g., 0.03 = 3%)
        trading_days: Trading days per year for annualized yield
        base_value: Starting value for TRI (default 100)

    Returns:
        DataFrame with: price_return, income_return, total_return, tri
    """
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")

    if yield_col not in df.columns:
        df[yield_col] = 0.0

    # Step 1: Daily income return from annualized yield
    df["income_return"] = df[yield_col] / trading_days

    # Step 2: Price return
    df["price_return"] = df[price_col].pct_change()

    # Step 3: Total return = price return + income return
    df["total_return"] = df["price_return"].fillna(0) + df["income_return"].fillna(0)

    # Step 4: Compound to total return index
    df["tri"] = (1 + df["total_return"]).cumprod()

    # Rebase to base_value
    first_valid = df["tri"].first_valid_index()
    if first_valid is not None:
        df["tri"] = base_value * df["tri"] / df.loc[first_valid, "tri"]

    return df
