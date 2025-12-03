"""
Number Cleaner Utilities
========================

Robust numeric parsing and cleaning for PRISM data pipelines.

Functions:
    - parse_numeric: Convert string to float with extensive handling
    - clean_numeric_column: Clean and normalize a numeric column in a DataFrame
    - is_numeric_value: Check if a value can be converted to numeric
    - handle_percentage: Convert percentage string to decimal
"""

from __future__ import annotations

import math
import re
from typing import Optional, Union

import numpy as np
import pandas as pd

# Common representations of missing/null values
NULL_VALUES = {
    "",
    ".",
    "-",
    "--",
    "n/a",
    "na",
    "nan",
    "null",
    "none",
    "nil",
    "#n/a",
    "#na",
    "#null",
    "#value!",
    "#ref!",
    "#div/0!",
    "missing",
    "unavailable",
    "nd",       # Not Determined
    "n.d.",
    "n.a.",
    "...",
    "—",        # Em dash
    "–",        # En dash
}

# Pattern to match percentage values
PERCENTAGE_PATTERN = re.compile(r'^([+-]?\d*\.?\d+)\s*%$')

# Pattern to match numbers with thousands separators
THOUSANDS_PATTERN = re.compile(r'^([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)$')

# Pattern to match scientific notation
SCIENTIFIC_PATTERN = re.compile(r'^([+-]?\d*\.?\d+)[eE]([+-]?\d+)$')

# Pattern to match currency values
CURRENCY_PATTERN = re.compile(r'^[$€£¥₹]?\s*([+-]?\d*,?\d*\.?\d+)\s*[KMBTkmbt]?$')


def parse_numeric(
    value: Union[str, int, float, None],
    handle_percent: bool = True,
    allow_infinity: bool = False,
) -> Optional[float]:
    """
    Convert a value to float with extensive handling of edge cases.

    Handles:
        - Commas as thousands separators
        - Percentage signs (converts to decimal if handle_percent=True)
        - Common null/NA representations
        - Scientific notation
        - Leading/trailing whitespace
        - Currency symbols

    Args:
        value: The value to convert
        handle_percent: If True, convert percentages to decimal (50% -> 0.5)
        allow_infinity: If True, allow inf values; otherwise return None

    Returns:
        Float value or None if conversion fails

    Examples:
        >>> parse_numeric("1,234.56")
        1234.56
        >>> parse_numeric("50%")
        0.5
        >>> parse_numeric("N/A")
        None
        >>> parse_numeric("$1,234.56")
        1234.56
    """
    # Handle None and non-string numerics
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        if math.isinf(value) and not allow_infinity:
            return None
        return float(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        if np.isinf(value) and not allow_infinity:
            return None
        return float(value)

    if isinstance(value, np.integer):
        return float(value)

    if not isinstance(value, str):
        return None

    # Clean up string
    cleaned = value.strip()

    # Check for null values
    if cleaned.lower() in NULL_VALUES:
        return None

    # Handle empty string
    if not cleaned:
        return None

    # Handle percentage
    if handle_percent:
        pct_match = PERCENTAGE_PATTERN.match(cleaned)
        if pct_match:
            try:
                return float(pct_match.group(1)) / 100.0
            except ValueError:
                return None

    # Remove currency symbols
    cleaned = re.sub(r'^[$€£¥₹]\s*', '', cleaned)
    cleaned = re.sub(r'\s*[$€£¥₹]$', '', cleaned)

    # Handle suffixes (K, M, B, T for thousands, millions, billions, trillions)
    multiplier = 1.0
    suffix_match = re.search(r'([KMBTkmbt])$', cleaned)
    if suffix_match:
        suffix = suffix_match.group(1).upper()
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
        multiplier = multipliers.get(suffix, 1.0)
        cleaned = cleaned[:-1].strip()

    # Remove thousands separators (commas)
    cleaned = cleaned.replace(',', '')

    # Remove spaces (some locales use space as thousands separator)
    cleaned = cleaned.replace(' ', '')

    # Handle parentheses for negative numbers: (123.45) -> -123.45
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]

    # Try direct conversion
    try:
        result = float(cleaned) * multiplier

        if math.isnan(result):
            return None

        if math.isinf(result) and not allow_infinity:
            return None

        return result

    except (ValueError, TypeError):
        return None


def is_numeric_value(value: Union[str, int, float, None]) -> bool:
    """
    Check if a value can be converted to a valid numeric.

    Args:
        value: The value to check

    Returns:
        True if value can be converted to float, False otherwise
    """
    return parse_numeric(value) is not None


def handle_percentage(
    value: Union[str, float, None],
    already_decimal: bool = False,
) -> Optional[float]:
    """
    Convert percentage to decimal form.

    Args:
        value: Percentage value (e.g., "50%" or 50.0)
        already_decimal: If True, value is already in decimal form (0.5 for 50%)

    Returns:
        Decimal representation (0.5 for 50%)
    """
    if value is None:
        return None

    if already_decimal:
        parsed = parse_numeric(value, handle_percent=False)
        return parsed

    if isinstance(value, str) and '%' in value:
        return parse_numeric(value, handle_percent=True)

    # Assume raw percentage value (e.g., 50.0 means 50%)
    parsed = parse_numeric(value, handle_percent=False)
    if parsed is not None:
        return parsed / 100.0

    return None


def clean_numeric_column(
    df: pd.DataFrame,
    column: str,
    handle_percent: bool = True,
    drop_invalid: bool = False,
    fill_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Clean and normalize a numeric column in a DataFrame.

    Args:
        df: Input DataFrame
        column: Name of the column to clean
        handle_percent: If True, convert percentages to decimal
        drop_invalid: If True, drop rows where column is invalid
        fill_value: Value to fill for invalid entries (ignored if drop_invalid=True)

    Returns:
        DataFrame with cleaned numeric column
    """
    if df.empty or column not in df.columns:
        return df

    df = df.copy()

    # Parse all values
    cleaned_values = df[column].apply(
        lambda x: parse_numeric(x, handle_percent=handle_percent)
    )

    if drop_invalid:
        # Keep only rows with valid values
        valid_mask = cleaned_values.notna()
        df = df[valid_mask].reset_index(drop=True)
        df[column] = cleaned_values[valid_mask].reset_index(drop=True)
    else:
        if fill_value is not None:
            cleaned_values = cleaned_values.fillna(fill_value)
        df[column] = cleaned_values

    return df


def coerce_to_float_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to float type, handling various input formats.

    Args:
        series: Input Series with potentially mixed types

    Returns:
        Series of float values (with NaN for invalid values)
    """
    return series.apply(lambda x: parse_numeric(x)).astype(float)


def detect_numeric_garbage(series: pd.Series, threshold: float = 0.1) -> bool:
    """
    Detect if a series contains too many non-numeric values (garbage).

    Args:
        series: Series to check
        threshold: Maximum allowed fraction of non-numeric values

    Returns:
        True if garbage detected (more than threshold fraction invalid)
    """
    if series.empty:
        return False

    invalid_count = series.apply(lambda x: not is_numeric_value(x)).sum()
    invalid_fraction = invalid_count / len(series)

    return invalid_fraction > threshold


def summarize_numeric_issues(series: pd.Series) -> dict:
    """
    Generate a summary of numeric parsing issues in a series.

    Args:
        series: Series to analyze

    Returns:
        Dictionary with counts of different issue types
    """
    issues = {
        "total_rows": len(series),
        "valid_numeric": 0,
        "null_values": 0,
        "unparseable": 0,
        "negative": 0,
        "zero": 0,
        "examples_unparseable": [],
    }

    unparseable_examples = []

    for val in series:
        parsed = parse_numeric(val)

        if parsed is None:
            if val is None or (isinstance(val, str) and val.strip().lower() in NULL_VALUES):
                issues["null_values"] += 1
            else:
                issues["unparseable"] += 1
                if len(unparseable_examples) < 5 and val not in unparseable_examples:
                    unparseable_examples.append(str(val))
        else:
            issues["valid_numeric"] += 1
            if parsed < 0:
                issues["negative"] += 1
            elif parsed == 0:
                issues["zero"] += 1

    issues["examples_unparseable"] = unparseable_examples

    return issues
