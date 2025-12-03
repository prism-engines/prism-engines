"""
PRISM Panel Validators
======================

Validation utilities for ensuring panel data quality.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PanelValidationError(Exception):
    """Raised when panel validation fails."""
    pass


def validate_panel(
    df: pd.DataFrame,
    min_rows: int = 100,
    max_nan_ratio: float = 0.5,
    raise_on_error: bool = False,
) -> dict:
    """
    Validate the constructed panel DataFrame.

    Performs the following checks:
    - Date index is sorted in ascending order
    - No duplicate dates
    - At least `min_rows` rows
    - Logs columns with too many NaNs (ratio > max_nan_ratio)

    Args:
        df: Panel DataFrame with date index and series columns
        min_rows: Minimum required number of rows
        max_nan_ratio: Maximum allowed ratio of NaN values per column
        raise_on_error: If True, raises PanelValidationError on validation failure

    Returns:
        Dictionary with validation results:
        - is_valid: bool - overall validation status
        - errors: list[str] - critical errors
        - warnings: list[str] - non-critical warnings
        - stats: dict - summary statistics

    Raises:
        PanelValidationError: If raise_on_error=True and validation fails
    """
    errors = []
    warnings = []
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "date_range": None,
        "sparse_columns": [],
    }

    # Check if DataFrame is empty
    if df.empty:
        errors.append("Panel DataFrame is empty")
        result = {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
        }
        if raise_on_error:
            raise PanelValidationError("Panel validation failed: " + "; ".join(errors))
        return result

    # Validate index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            errors.append("Index cannot be converted to DatetimeIndex")

    # Check date index is sorted
    if not df.index.is_monotonic_increasing:
        errors.append("Date index is not sorted in ascending order")
        logger.error("Panel validation: Date index is not sorted")

    # Check for duplicate dates
    duplicates = df.index.duplicated()
    if duplicates.any():
        num_dups = duplicates.sum()
        errors.append(f"Found {num_dups} duplicate date(s) in index")
        logger.error(f"Panel validation: {num_dups} duplicate dates found")

    # Check minimum row count
    if len(df) < min_rows:
        errors.append(
            f"Insufficient rows: {len(df)} < {min_rows} minimum required"
        )
        logger.error(
            f"Panel validation: Only {len(df)} rows, need at least {min_rows}"
        )

    # Record date range
    if len(df) > 0 and isinstance(df.index, pd.DatetimeIndex):
        stats["date_range"] = {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
        }

    # Check NaN ratios per column
    sparse_columns = []
    for col in df.columns:
        nan_ratio = df[col].isna().mean()
        if nan_ratio > max_nan_ratio:
            sparse_columns.append({
                "column": col,
                "nan_ratio": round(nan_ratio, 4),
            })
            warnings.append(
                f"Column '{col}' has {nan_ratio:.1%} NaN values "
                f"(threshold: {max_nan_ratio:.1%})"
            )
            logger.warning(
                f"Panel validation: Column '{col}' is sparse "
                f"({nan_ratio:.1%} NaN)"
            )

    stats["sparse_columns"] = sparse_columns

    # Determine overall validity
    is_valid = len(errors) == 0

    result = {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }

    if is_valid:
        logger.info(
            f"Panel validation passed: {len(df)} rows, "
            f"{len(df.columns)} columns"
        )
    else:
        logger.error(
            f"Panel validation failed with {len(errors)} error(s)"
        )

    if raise_on_error and not is_valid:
        raise PanelValidationError(
            "Panel validation failed: " + "; ".join(errors)
        )

    return result


def validate_registry_entry(
    entry: dict,
    registry_type: str,
) -> tuple[bool, list[str]]:
    """
    Validate a single registry entry.

    Args:
        entry: Dictionary containing registry entry data
        registry_type: Type of registry ('market' or 'economic')

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Common required fields
    required_common = ["fetch_type", "enabled", "frequency", "use_column"]
    for field in required_common:
        if field not in entry:
            errors.append(f"Missing required field: {field}")

    # Market-specific validation
    if registry_type == "market":
        if "tables" not in entry:
            errors.append("Market entry missing 'tables' mapping")

    # Economic-specific validation
    if registry_type == "economic":
        if "table" not in entry:
            errors.append("Economic entry missing 'table' field")
        if "code" not in entry:
            errors.append("Economic entry missing 'code' field")

    # Validate frequency values
    valid_frequencies = ["daily", "weekly", "monthly", "quarterly", "yearly"]
    if "frequency" in entry and entry["frequency"] not in valid_frequencies:
        errors.append(
            f"Invalid frequency '{entry['frequency']}'. "
            f"Valid values: {valid_frequencies}"
        )

    return len(errors) == 0, errors


def check_column_consistency(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> dict:
    """
    Check if panel has expected columns.

    Args:
        df: Panel DataFrame
        expected_columns: List of expected column names

    Returns:
        Dictionary with:
        - missing: columns expected but not found
        - extra: columns found but not expected
        - present: columns that match
    """
    actual = set(df.columns)
    expected = set(expected_columns)

    return {
        "missing": list(expected - actual),
        "extra": list(actual - expected),
        "present": list(expected & actual),
    }
