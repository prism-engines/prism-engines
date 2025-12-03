"""
Date Cleaner Utilities
======================

Robust date parsing and cleaning for PRISM data pipelines.

Functions:
    - parse_date_strict: Parse date string to datetime/date with strict validation
    - fix_two_digit_year: Convert 2-digit years to 4-digit
    - strip_time_and_timezone: Remove time components from datetime
    - validate_date_range: Check if date is within valid range
    - clean_date_column: Clean and normalize a date column in a DataFrame
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd

# Common date formats to try (ordered by likelihood)
DATE_FORMATS = [
    "%Y-%m-%d",       # ISO standard: 2024-01-15
    "%Y/%m/%d",       # 2024/01/15
    "%m/%d/%Y",       # US format: 01/15/2024
    "%d/%m/%Y",       # EU format: 15/01/2024
    "%m-%d-%Y",       # 01-15-2024
    "%d-%m-%Y",       # 15-01-2024
    "%Y%m%d",         # Compact: 20240115
    "%b %d, %Y",      # Jan 15, 2024
    "%B %d, %Y",      # January 15, 2024
    "%d %b %Y",       # 15 Jan 2024
    "%d %B %Y",       # 15 January 2024
    "%Y-%m-%dT%H:%M:%S",  # ISO with time
    "%Y-%m-%d %H:%M:%S",  # Date with time
]

# Cutoff for 2-digit year interpretation (years below this are 2000s, above are 1900s)
TWO_DIGIT_YEAR_CUTOFF = 30


def parse_date_strict(
    date_str: Union[str, datetime, date, pd.Timestamp],
    reject_future: bool = True,
    max_past_year: int = 1900,
) -> Optional[date]:
    """
    Parse a date string to a date object with strict validation.

    Args:
        date_str: The date string or datetime-like object to parse
        reject_future: If True, reject dates in the future
        max_past_year: Reject dates before this year

    Returns:
        A date object if parsing succeeds, None if invalid

    Examples:
        >>> parse_date_strict("2024-01-15")
        datetime.date(2024, 1, 15)
        >>> parse_date_strict("01/15/2024")
        datetime.date(2024, 1, 15)
        >>> parse_date_strict("invalid")
        None
    """
    if date_str is None:
        return None

    # Handle already-parsed types
    if isinstance(date_str, datetime):
        result = date_str.date()
    elif isinstance(date_str, date):
        result = date_str
    elif isinstance(date_str, pd.Timestamp):
        result = date_str.date()
    elif isinstance(date_str, str):
        result = _parse_date_string(date_str)
    else:
        return None

    if result is None:
        return None

    # Validate range
    today = date.today()

    if reject_future and result > today:
        return None

    if result.year < max_past_year:
        return None

    return result


def _parse_date_string(date_str: str) -> Optional[date]:
    """
    Internal helper to parse a date string using multiple formats.

    Args:
        date_str: The date string to parse

    Returns:
        A date object if parsing succeeds, None otherwise
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # Clean up the string
    date_str = date_str.strip()

    # Handle empty strings
    if not date_str:
        return None

    # Try pandas first (handles many formats)
    try:
        parsed = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(parsed):
            return parsed.date()
    except (ValueError, TypeError):
        pass

    # Try explicit formats
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.date()
        except ValueError:
            continue

    return None


def fix_two_digit_year(date_str: str) -> str:
    """
    Convert 2-digit years to 4-digit years.

    Uses cutoff logic: years < 30 become 20xx, years >= 30 become 19xx.

    Args:
        date_str: Date string that may contain 2-digit year

    Returns:
        Date string with 4-digit year

    Examples:
        >>> fix_two_digit_year("01/15/24")
        "01/15/2024"
        >>> fix_two_digit_year("01/15/95")
        "01/15/1995"
    """
    if not date_str or not isinstance(date_str, str):
        return date_str

    # Pattern to find 2-digit years at typical positions
    # Matches: /YY, -YY, or YY at end of string after separator
    patterns = [
        (r'/(\d{2})$', '/'),       # 01/15/24 -> 01/15/2024
        (r'-(\d{2})$', '-'),       # 01-15-24 -> 01-15-2024
        (r'^(\d{2})[/-]', ''),     # 24-01-15 -> 2024-01-15
    ]

    for pattern, sep in patterns:
        match = re.search(pattern, date_str)
        if match:
            year_2d = int(match.group(1))
            if year_2d < TWO_DIGIT_YEAR_CUTOFF:
                year_4d = 2000 + year_2d
            else:
                year_4d = 1900 + year_2d

            if sep:
                date_str = re.sub(pattern, f'{sep}{year_4d}', date_str)
            else:
                date_str = re.sub(pattern, f'{year_4d}{date_str[2]}', date_str)
            break

    return date_str


def strip_time_and_timezone(
    dt: Union[datetime, pd.Timestamp, str]
) -> Optional[date]:
    """
    Remove time and timezone components, returning just the date.

    Args:
        dt: A datetime object, Timestamp, or string

    Returns:
        A date object with only year, month, day

    Examples:
        >>> strip_time_and_timezone("2024-01-15T10:30:00Z")
        datetime.date(2024, 1, 15)
    """
    if dt is None:
        return None

    if isinstance(dt, str):
        parsed = parse_date_strict(dt, reject_future=False)
        return parsed

    if isinstance(dt, datetime):
        return dt.date()

    if isinstance(dt, pd.Timestamp):
        return dt.date()

    if isinstance(dt, date):
        return dt

    return None


def validate_date_range(
    dt: date,
    min_date: Optional[date] = None,
    max_date: Optional[date] = None,
) -> bool:
    """
    Check if a date falls within the specified range.

    Args:
        dt: The date to validate
        min_date: Minimum allowed date (inclusive)
        max_date: Maximum allowed date (inclusive)

    Returns:
        True if date is within range, False otherwise
    """
    if dt is None:
        return False

    if min_date and dt < min_date:
        return False

    if max_date and dt > max_date:
        return False

    return True


def clean_date_column(
    df: pd.DataFrame,
    date_column: str = "date",
    output_format: str = "%Y-%m-%d",
    reject_future: bool = True,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """
    Clean and normalize a date column in a DataFrame.

    Args:
        df: Input DataFrame
        date_column: Name of the date column
        output_format: Output date format (ISO standard by default)
        reject_future: If True, mark future dates as invalid
        drop_invalid: If True, drop rows with invalid dates

    Returns:
        DataFrame with cleaned date column
    """
    if df.empty or date_column not in df.columns:
        return df

    df = df.copy()
    today = date.today()

    # Parse and clean dates
    cleaned_dates = []
    valid_mask = []

    for val in df[date_column]:
        parsed = parse_date_strict(val, reject_future=reject_future)
        if parsed is not None:
            cleaned_dates.append(parsed.strftime(output_format))
            valid_mask.append(True)
        else:
            cleaned_dates.append(None)
            valid_mask.append(False)

    df[date_column] = cleaned_dates

    if drop_invalid:
        df = df[valid_mask].reset_index(drop=True)

    return df


def to_iso_date(dt: Union[str, datetime, date, pd.Timestamp, None]) -> Optional[str]:
    """
    Convert any date representation to ISO format string (YYYY-MM-DD).

    Args:
        dt: Date in any supported format

    Returns:
        ISO format date string or None if invalid
    """
    parsed = parse_date_strict(dt, reject_future=False)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%d")
