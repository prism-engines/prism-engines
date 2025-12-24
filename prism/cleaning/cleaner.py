"""
PRISM Data Cleaner

Cleans raw indicator data for engine consumption.

Cleaning rules are documented in /docs/cleaning_rules.md
This module implements those rules programmatically.

Architecture:
    Fetcher → Cleaner → DB (raw + clean tables)
    
The cleaner:
    - Receives raw DataFrame from fetcher
    - Returns cleaned DataFrame
    - Does NOT normalize (engines handle that)
    - Records cleaning method applied
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class CleanResult:
    """Result of cleaning operation."""
    data: pd.DataFrame
    method: str
    rows_before: int
    rows_after: int
    nulls_filled: int
    
    @property
    def rows_dropped(self) -> int:
        return self.rows_before - self.rows_after


def clean_series(
    df: pd.DataFrame,
    frequency: Optional[str] = None,
    max_gap_fill: int = 5,
) -> CleanResult:

    """
    Clean a raw indicator DataFrame.
    
    Cleaning steps:
        1. Ensure date column is datetime
        2. Sort by date
        3. Remove duplicate dates (keep last)
        4. Forward-fill small gaps (up to max_gap_fill)
        5. Drop remaining NaN values
    
    Args:
        df: Raw DataFrame with columns [date, value]
        frequency: Indicator frequency ('daily', 'weekly', 'monthly', 'quarterly')
                   Used to determine appropriate gap handling.
        max_gap_fill: Maximum consecutive NaN values to forward-fill.
                      Default 5 (covers weekends + a few holidays for daily data).
    
    Returns:
        CleanResult with cleaned DataFrame and metadata
    """
    if df.empty:
        return CleanResult(
            data=df.copy(),
            method="none_empty_input",
            rows_before=0,
            rows_after=0,
            nulls_filled=0,
        )
    
    rows_before = len(df)
    clean_df = df.copy()
    
    # Step 1: Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(clean_df["date"]):
        clean_df["date"] = pd.to_datetime(clean_df["date"])
    
    # Normalize to date only (remove time component)
    clean_df["date"] = clean_df["date"].dt.date
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    
    # Step 2: Sort by date
    clean_df = clean_df.sort_values("date").reset_index(drop=True)
    
    # Step 3: Remove duplicate dates (keep last value)
    dupes = clean_df.duplicated(subset=["date"], keep="last").sum()
    if dupes > 0:
        logger.debug(f"Removing {dupes} duplicate dates")
        clean_df = clean_df.drop_duplicates(subset=["date"], keep="last")
    
    # Step 4: Count nulls before filling
    nulls_before = clean_df["value"].isna().sum()
    
    # Step 5: Forward-fill small gaps
    # Use limit to prevent filling large gaps
    clean_df["value"] = clean_df["value"].ffill(limit=max_gap_fill)
    
    nulls_after = clean_df["value"].isna().sum()
    nulls_filled = nulls_before - nulls_after
    
    # Step 6: Drop remaining NaN values
    clean_df = clean_df.dropna(subset=["value"])
    
    rows_after = len(clean_df)
    
    # Build method string
    method_parts = ["sorted", "deduped"]
    if nulls_filled > 0:
        method_parts.append(f"ffill_{max_gap_fill}")
    if rows_before - rows_after > 0:
        method_parts.append("dropped_nulls")
    method = "_".join(method_parts)
    
    logger.debug(
        f"Cleaned: {rows_before} → {rows_after} rows, "
        f"filled {nulls_filled} nulls, method={method}"
    )
    
    return CleanResult(
        data=clean_df,
        method=method,
        rows_before=rows_before,
        rows_after=rows_after,
        nulls_filled=nulls_filled,
    )


def align_to_daily(
    df: pd.DataFrame,
    frequency: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Align non-daily data to daily frequency via forward-fill.
    
    Monthly/quarterly values are applied to all days in the period.
    
    Args:
        df: Cleaned DataFrame with columns [date, value]
        frequency: Source frequency ('weekly', 'monthly', 'quarterly')
        start_date: Optional start date for daily range
        end_date: Optional end date for daily range
    
    Returns:
        DataFrame with daily frequency
    """
    if frequency == "daily" or df.empty:
        return df
    
    # Create daily date range
    if start_date is None:
        start_date = df["date"].min()
    if end_date is None:
        end_date = df["date"].max()
    
    daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    daily_df = pd.DataFrame({"date": daily_dates})
    
    # Merge and forward-fill
    merged = daily_df.merge(df, on="date", how="left")
    merged["value"] = merged["value"].ffill()
    
    # Drop leading NaNs (before first data point)
    merged = merged.dropna(subset=["value"])
    
    return merged


def validate_clean_data(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate that cleaned data meets requirements.
    
    Checks:
        - Has required columns
        - No NaN values
        - Dates are sorted
        - No duplicate dates
    
    Returns:
        (is_valid, error_message)
    """
    # Check columns
    required = {"date", "value"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        return False, f"Missing columns: {missing}"
    
    # Check for NaN
    if df["value"].isna().any():
        null_count = df["value"].isna().sum()
        return False, f"Contains {null_count} NaN values"
    
    # Check sorted
    if not df["date"].is_monotonic_increasing:
        return False, "Dates not sorted"
    
    # Check duplicates
    if df["date"].duplicated().any():
        dupe_count = df["date"].duplicated().sum()
        return False, f"Contains {dupe_count} duplicate dates"
    
    return True, None
