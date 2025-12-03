"""

Validation and cleaning functions for fetched data before DB insertion.

This module provides the "defense layer" against garbage data entering
the PRISM database.

Functions:
    - validate_dataframe_shape: Check minimum rows, column requirements
    - detect_footer_garbage: Find and flag trailing non-data rows
    - remove_footer_garbage: Remove trailing junk rows
    - validate_no_duplicate_dates: Ensure unique dates per series
    - validate_date_sequence: Ensure dates are properly ordered
    - validate_frequency: Check if data frequency matches expected
    - validate_numeric_columns: Ensure numeric columns have valid types
    - comprehensive_validate: Run all validations and return report
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .date_cleaner import parse_date_strict, to_iso_date
from .number_cleaner import is_numeric_value, parse_numeric

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails critically."""
    pass


class ValidationWarning:
    """Container for validation warnings (non-critical issues)."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}

    def __str__(self):
        return f"ValidationWarning: {self.message}"


def validate_dataframe_shape(
    df: pd.DataFrame,
    min_rows: int = 1,
    required_columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame shape requirements.

    Args:
        df: DataFrame to validate
        min_rows: Minimum required rows
        required_columns: List of required column names
        max_rows: Maximum allowed rows (None for unlimited)

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if df is None:
        return False, ["DataFrame is None"]

    if df.empty:
        return False, ["DataFrame is empty"]

    if len(df) < min_rows:
        errors.append(f"Insufficient rows: {len(df)} < {min_rows}")

    if max_rows and len(df) > max_rows:
        errors.append(f"Too many rows: {len(df)} > {max_rows}")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")

    return len(errors) == 0, errors


def detect_footer_garbage(
    df: pd.DataFrame,
    date_column: str = "date",
    value_columns: Optional[List[str]] = None,
    tail_rows: int = 10,
) -> List[int]:
    """
    Detect trailing rows that appear to be garbage (footers, notes, etc.).

    Garbage rows typically have:
    - Invalid dates
    - Non-numeric values in numeric columns
    - Repeated patterns indicating source metadata

    Args:
        df: DataFrame to check
        date_column: Name of date column
        value_columns: Names of numeric columns to check
        tail_rows: Number of trailing rows to inspect

    Returns:
        List of row indices that appear to be garbage
    """
    if df.empty:
        return []

    garbage_indices = []

    # Check the last N rows
    start_idx = max(0, len(df) - tail_rows)

    for idx in range(len(df) - 1, start_idx - 1, -1):
        row = df.iloc[idx]

        is_garbage = False

        # Check date column
        if date_column in df.columns:
            date_val = row[date_column]
            parsed = parse_date_strict(date_val, reject_future=False)
            if parsed is None:
                is_garbage = True

        # Check value columns
        if value_columns and not is_garbage:
            for col in value_columns:
                if col in df.columns:
                    val = row[col]
                    if not is_numeric_value(val):
                        # Check if it looks like a footer (contains letters)
                        if isinstance(val, str) and any(c.isalpha() for c in val):
                            is_garbage = True
                            break

        if is_garbage:
            garbage_indices.append(idx)
        else:
            # Stop when we hit valid data (garbage is typically at the end)
            break

    return sorted(garbage_indices)


def remove_footer_garbage(
    df: pd.DataFrame,
    date_column: str = "date",
    value_columns: Optional[List[str]] = None,
    log_removed: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove trailing garbage rows from DataFrame.

    Args:
        df: DataFrame to clean
        date_column: Name of date column
        value_columns: Names of numeric columns to check
        log_removed: If True, log information about removed rows

    Returns:
        Tuple of (cleaned DataFrame, number of rows removed)
    """
    garbage_indices = detect_footer_garbage(df, date_column, value_columns)

    if not garbage_indices:
        return df, 0

    if log_removed:
        logger.warning(
            f"Removing {len(garbage_indices)} footer garbage rows "
            f"(indices {min(garbage_indices)}-{max(garbage_indices)})"
        )

        # Log sample of garbage
        for idx in garbage_indices[:3]:
            logger.debug(f"  Garbage row {idx}: {df.iloc[idx].to_dict()}")

    # Remove garbage rows
    df_clean = df.drop(index=garbage_indices).reset_index(drop=True)

    return df_clean, len(garbage_indices)


def validate_no_duplicate_dates(
    df: pd.DataFrame,
    date_column: str = "date",
    group_columns: Optional[List[str]] = None,
) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validate that there are no duplicate dates in the DataFrame.

    Args:
        df: DataFrame to validate
        date_column: Name of date column
        group_columns: Additional columns that form a unique key with date

    Returns:
        Tuple of (is_valid, error messages, DataFrame of duplicates)
    """
    if df.empty or date_column not in df.columns:
        return True, [], pd.DataFrame()

    # Build the key columns
    key_cols = [date_column]
    if group_columns:
        key_cols.extend([c for c in group_columns if c in df.columns])

    # Find duplicates
    duplicates = df[df.duplicated(subset=key_cols, keep=False)]

    if duplicates.empty:
        return True, [], pd.DataFrame()

    n_dups = len(duplicates)
    unique_dates = duplicates[date_column].nunique()

    errors = [
        f"Found {n_dups} duplicate rows across {unique_dates} dates"
    ]

    return False, errors, duplicates


def remove_duplicate_dates(
    df: pd.DataFrame,
    date_column: str = "date",
    keep: str = "last",
    group_columns: Optional[List[str]] = None,
    log_removed: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate date rows, keeping first or last occurrence.

    Args:
        df: DataFrame to clean
        date_column: Name of date column
        keep: Which duplicate to keep ('first', 'last')
        group_columns: Additional columns for uniqueness check
        log_removed: If True, log removed duplicates

    Returns:
        Tuple of (cleaned DataFrame, number of rows removed)
    """
    if df.empty or date_column not in df.columns:
        return df, 0

    key_cols = [date_column]
    if group_columns:
        key_cols.extend([c for c in group_columns if c in df.columns])

    n_before = len(df)
    df_clean = df.drop_duplicates(subset=key_cols, keep=keep).reset_index(drop=True)
    n_removed = n_before - len(df_clean)

    if log_removed and n_removed > 0:
        logger.warning(f"Removed {n_removed} duplicate date rows (kept {keep})")

    return df_clean, n_removed


def validate_date_sequence(
    df: pd.DataFrame,
    date_column: str = "date",
    ascending: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate that dates are in proper sequential order.

    Args:
        df: DataFrame to validate
        date_column: Name of date column
        ascending: If True, expect ascending order; if False, descending

    Returns:
        Tuple of (is_valid, error messages)
    """
    if df.empty or date_column not in df.columns:
        return True, []

    dates = pd.to_datetime(df[date_column], errors='coerce')

    # Check for out-of-order dates
    if ascending:
        out_of_order = dates.diff().lt(timedelta(0)).sum()
    else:
        out_of_order = dates.diff().gt(timedelta(0)).sum()

    if out_of_order > 0:
        direction = "ascending" if ascending else "descending"
        return False, [f"Found {out_of_order} dates out of {direction} order"]

    return True, []


def sort_by_date(
    df: pd.DataFrame,
    date_column: str = "date",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Sort DataFrame by date column.

    Args:
        df: DataFrame to sort
        date_column: Name of date column
        ascending: Sort order

    Returns:
        Sorted DataFrame
    """
    if df.empty or date_column not in df.columns:
        return df

    df = df.copy()
    df["_sort_date"] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.sort_values("_sort_date", ascending=ascending)
    df = df.drop(columns=["_sort_date"]).reset_index(drop=True)

    return df


def validate_frequency(
    df: pd.DataFrame,
    date_column: str = "date",
    expected_frequency: str = "daily",
    tolerance: float = 0.2,
) -> Tuple[bool, List[str]]:
    """
    Validate that data frequency roughly matches expected frequency.

    Args:
        df: DataFrame to validate
        date_column: Name of date column
        expected_frequency: Expected frequency ('daily', 'weekly', 'monthly', 'quarterly')
        tolerance: Allowed deviation from expected as fraction

    Returns:
        Tuple of (is_valid, warning messages)
    """
    if len(df) < 2 or date_column not in df.columns:
        return True, []

    dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
    if len(dates) < 2:
        return True, []

    # Calculate median gap
    gaps = dates.diff().dropna()
    median_gap_days = gaps.median().days

    # Expected gaps
    expected_gaps = {
        "daily": 1,
        "weekly": 7,
        "monthly": 30,
        "quarterly": 91,
        "yearly": 365,
    }

    expected_gap = expected_gaps.get(expected_frequency.lower(), 1)

    # Check if within tolerance
    lower_bound = expected_gap * (1 - tolerance)
    upper_bound = expected_gap * (1 + tolerance)

    # For daily data, also allow for weekends/holidays (up to 3-4 day gaps)
    if expected_frequency.lower() == "daily":
        upper_bound = max(upper_bound, 4)

    if median_gap_days < lower_bound or median_gap_days > upper_bound:
        return False, [
            f"Data frequency mismatch: median gap is {median_gap_days:.1f} days, "
            f"expected ~{expected_gap} days for {expected_frequency} data"
        ]

    return True, []


def validate_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    allow_nan: bool = True,
    max_nan_fraction: float = 0.5,
) -> Tuple[bool, List[str]]:
    """
    Validate that specified columns contain numeric data.

    Args:
        df: DataFrame to validate
        columns: List of column names to validate
        allow_nan: If True, allow NaN values
        max_nan_fraction: Maximum allowed fraction of NaN values

    Returns:
        Tuple of (is_valid, error messages)
    """
    errors = []

    for col in columns:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found")
            continue

        # Try to convert to numeric
        numeric_series = df[col].apply(parse_numeric)

        # Check for non-numeric values
        nan_count = numeric_series.isna().sum()
        nan_fraction = nan_count / len(df) if len(df) > 0 else 0

        if not allow_nan and nan_count > 0:
            errors.append(f"Column '{col}' has {nan_count} non-numeric values")
        elif nan_fraction > max_nan_fraction:
            errors.append(
                f"Column '{col}' has too many non-numeric values: "
                f"{nan_fraction:.1%} > {max_nan_fraction:.1%}"
            )

    return len(errors) == 0, errors


def validate_no_future_dates(
    df: pd.DataFrame,
    date_column: str = "date",
) -> Tuple[bool, List[str], List[int]]:
    """
    Validate that no dates are in the future.

    Args:
        df: DataFrame to validate
        date_column: Name of date column

    Returns:
        Tuple of (is_valid, error messages, indices of future dates)
    """
    if df.empty or date_column not in df.columns:
        return True, [], []

    today = date.today()
    future_indices = []

    for idx, row in df.iterrows():
        parsed = parse_date_strict(row[date_column], reject_future=False)
        if parsed and parsed > today:
            future_indices.append(idx)

    if future_indices:
        return False, [f"Found {len(future_indices)} future dates"], future_indices

    return True, [], []


def comprehensive_validate(
    df: pd.DataFrame,
    date_column: str = "date",
    value_columns: Optional[List[str]] = None,
    expected_frequency: str = "daily",
    min_rows: int = 10,
    fix_issues: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run comprehensive validation and optionally fix issues.

    Args:
        df: DataFrame to validate
        date_column: Name of date column
        value_columns: Names of numeric value columns
        expected_frequency: Expected data frequency
        min_rows: Minimum required rows
        fix_issues: If True, attempt to fix found issues

    Returns:
        Tuple of (validated/cleaned DataFrame, validation report dict)
    """
    report = {
        "original_rows": len(df),
        "final_rows": 0,
        "valid": True,
        "errors": [],
        "warnings": [],
        "fixes_applied": [],
        "rows_removed": 0,
    }

    if df.empty:
        report["valid"] = False
        report["errors"].append("DataFrame is empty")
        return df, report

    df_clean = df.copy()

    # 1. Check shape
    valid, errors = validate_dataframe_shape(
        df_clean,
        min_rows=min_rows,
        required_columns=[date_column],
    )
    if not valid:
        report["errors"].extend(errors)
        if not fix_issues:
            report["valid"] = False
            return df_clean, report

    # 2. Remove footer garbage
    if fix_issues:
        df_clean, n_removed = remove_footer_garbage(
            df_clean, date_column, value_columns
        )
        if n_removed > 0:
            report["fixes_applied"].append(f"Removed {n_removed} footer garbage rows")
            report["rows_removed"] += n_removed

    # 3. Validate and fix dates
    valid, errors, future_indices = validate_no_future_dates(df_clean, date_column)
    if not valid:
        if fix_issues:
            df_clean = df_clean.drop(index=future_indices).reset_index(drop=True)
            report["fixes_applied"].append(f"Removed {len(future_indices)} future dates")
            report["rows_removed"] += len(future_indices)
        else:
            report["errors"].extend(errors)

    # 4. Remove duplicate dates
    valid, errors, dups = validate_no_duplicate_dates(df_clean, date_column)
    if not valid:
        if fix_issues:
            df_clean, n_removed = remove_duplicate_dates(df_clean, date_column)
            report["fixes_applied"].append(f"Removed {n_removed} duplicate dates")
            report["rows_removed"] += n_removed
        else:
            report["errors"].extend(errors)

    # 5. Sort by date
    valid, errors = validate_date_sequence(df_clean, date_column)
    if not valid:
        if fix_issues:
            df_clean = sort_by_date(df_clean, date_column)
            report["fixes_applied"].append("Sorted by date ascending")
        else:
            report["warnings"].extend(errors)

    # 6. Check frequency
    valid, warnings = validate_frequency(df_clean, date_column, expected_frequency)
    if not valid:
        report["warnings"].extend(warnings)

    # 7. Validate numeric columns
    if value_columns:
        valid, errors = validate_numeric_columns(df_clean, value_columns)
        if not valid:
            report["warnings"].extend(errors)

    # Final check
    if len(df_clean) < min_rows:
        report["valid"] = False
        report["errors"].append(
            f"Insufficient rows after cleaning: {len(df_clean)} < {min_rows}"
        )
    else:
        report["valid"] = len(report["errors"]) == 0

    report["final_rows"] = len(df_clean)

    return df_clean, report
