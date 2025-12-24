"""
Data loading utilities for PRISM Engines.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional


def load_csv(
    path: Union[str, Path],
    index_col: Optional[Union[str, int]] = 0,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV file for PRISM analysis.

    Args:
        path: Path to CSV file
        index_col: Column to use as index (default: first column)
        parse_dates: Whether to parse dates (default: True)

    Returns:
        DataFrame with numeric columns ready for analysis

    Example:
        >>> df = load_csv("prices.csv")
        >>> df.head()
                     AAPL    GOOGL     MSFT
        date
        2024-01-01  185.2   140.5    375.2
        2024-01-02  186.1   141.2    376.8
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.suffix.lower() == ".csv":
        raise ValueError(f"Expected .csv file, got: {path.suffix}")

    df = pd.read_csv(
        path,
        index_col=index_col,
        parse_dates=parse_dates if index_col is not None else False,
    )

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        raise ValueError("No numeric columns found in CSV")

    # Drop columns that are all NaN
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # Forward-fill then back-fill remaining NaN
    numeric_df = numeric_df.ffill().bfill()

    return numeric_df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate a DataFrame for PRISM analysis.

    Returns:
        dict with validation results
    """
    issues = []

    if df.empty:
        issues.append("DataFrame is empty")

    if len(df.columns) < 2:
        issues.append(f"Need at least 2 columns, got {len(df.columns)}")

    if len(df) < 10:
        issues.append(f"Need at least 10 rows, got {len(df)}")

    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        issues.append(f"Columns with NaN: {nan_cols}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "shape": df.shape,
        "columns": list(df.columns),
        "date_range": (
            str(df.index.min()),
            str(df.index.max()),
        ) if hasattr(df.index, 'min') else None,
    }
