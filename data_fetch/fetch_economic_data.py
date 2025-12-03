"""
Economic Data Fetcher
=====================

Registry-driven economic data fetching with strict normalization and validation.

This module fetches economic time series data (primarily from FRED)
based on the economic_registry.json configuration.

Key Features:
    - Registry-driven: All series defined in economic_registry.json
    - Strict normalization: ISO dates, float values, no garbage
    - Validation: No duplicates, proper frequency, sorted ascending
    - Revision handling: Support for economic data revisions
    - Database integration: Passes clean DataFrames to DB writer

Usage:
    from data_fetch.fetch_economic_data import fetch_all_economic_data

    # Fetch all enabled series
    results = fetch_all_economic_data()

    # Fetch single series
    df = fetch_single_economic_series("cpi")
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Import path setup
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.date_cleaner import clean_date_column, to_iso_date
from utils.number_cleaner import clean_numeric_column, parse_numeric
from utils.fetch_validator import (
    comprehensive_validate,
    remove_duplicate_dates,
    remove_footer_garbage,
    sort_by_date,
    validate_dataframe_shape,
    validate_frequency,
)

logger = logging.getLogger(__name__)

# Registry paths
REGISTRY_DIR = Path(__file__).parent.parent / "data" / "registry"
ECONOMIC_REGISTRY_PATH = REGISTRY_DIR / "economic_registry.json"
SYSTEM_REGISTRY_PATH = REGISTRY_DIR / "system_registry.json"


class EconomicFetchError(Exception):
    """Raised when economic data fetch fails."""
    pass


def load_system_registry() -> Dict[str, Any]:
    """
    Load the system registry configuration.

    Returns:
        Dictionary with system configuration
    """
    if not SYSTEM_REGISTRY_PATH.exists():
        logger.warning(f"System registry not found: {SYSTEM_REGISTRY_PATH}")
        return {}

    with open(SYSTEM_REGISTRY_PATH, "r") as f:
        return json.load(f)


def load_economic_registry() -> Dict[str, Any]:
    """
    Load the economic registry configuration.

    Returns:
        Dictionary with economic series configuration
    """
    if not ECONOMIC_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Economic registry not found: {ECONOMIC_REGISTRY_PATH}")

    with open(ECONOMIC_REGISTRY_PATH, "r") as f:
        return json.load(f)


def get_enabled_series(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get list of enabled series from registry.

    Args:
        registry: Economic registry dictionary

    Returns:
        List of enabled series configurations
    """
    series = registry.get("series", [])
    return [s for s in series if s.get("enabled", True)]


def normalize_economic_dataframe(
    df: pd.DataFrame,
    ticker: str,
    series_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize economic data DataFrame to standard format.

    Normalization includes:
        - Date column: ISO format YYYY-MM-DD
        - Value column: float values (renamed to value_raw)
        - Remove duplicates, sort ascending, remove garbage
        - Handle revision dates if applicable

    Args:
        df: Raw DataFrame from fetcher
        ticker: Series ticker
        series_config: Series configuration from registry

    Returns:
        Tuple of (normalized DataFrame, normalization report)
    """
    report = {
        "ticker": ticker,
        "original_rows": len(df),
        "final_rows": 0,
        "issues": [],
        "fixes": [],
    }

    if df is None or df.empty:
        report["issues"].append("Empty DataFrame received")
        return pd.DataFrame(), report

    df = df.copy()

    # Identify columns
    date_col = "date"
    value_col = None

    # Find the value column (usually named after ticker)
    ticker_lower = ticker.lower()
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == ticker_lower or col_lower == "value":
            value_col = col
            break

    if value_col is None:
        # Try to find any numeric column that's not date
        for col in df.columns:
            if col.lower() != "date":
                value_col = col
                break

    if value_col is None:
        report["issues"].append("Could not identify value column")
        return pd.DataFrame(), report

    # 1. Clean date column
    if date_col not in df.columns:
        report["issues"].append("No date column found")
        return pd.DataFrame(), report

    df = clean_date_column(df, date_col, drop_invalid=True)
    rows_after_date = len(df)
    if rows_after_date < report["original_rows"]:
        dropped = report["original_rows"] - rows_after_date
        report["fixes"].append(f"Dropped {dropped} rows with invalid dates")

    if df.empty:
        report["issues"].append("All rows had invalid dates")
        return pd.DataFrame(), report

    # 2. Remove footer garbage
    df, n_garbage = remove_footer_garbage(df, date_col, [value_col], log_removed=False)
    if n_garbage > 0:
        report["fixes"].append(f"Removed {n_garbage} footer garbage rows")

    # 3. Clean numeric column
    df = clean_numeric_column(df, value_col, drop_invalid=False)  # Keep NaN for economic data

    # 4. Handle duplicate dates
    # For economic data with revisions, we might want to keep the latest revision
    # For now, keep the last value for each date
    df, n_dups = remove_duplicate_dates(df, date_col, keep="last", log_removed=False)
    if n_dups > 0:
        report["fixes"].append(f"Removed {n_dups} duplicate dates (kept latest)")

    # 5. Sort by date ascending
    df = sort_by_date(df, date_col, ascending=True)

    # 6. Remove future dates
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    before_future = len(df)
    df = df[df[date_col] <= today_str].reset_index(drop=True)
    n_future = before_future - len(df)
    if n_future > 0:
        report["fixes"].append(f"Removed {n_future} future dates")

    # 7. Standardize column names
    df = df.rename(columns={
        date_col: "date",
        value_col: "value_raw",
    })

    # Select final columns
    final_cols = ["date", "value_raw"]
    df = df[final_cols]

    # 8. Validate frequency
    expected_freq = series_config.get("frequency", "monthly")
    valid, warnings = validate_frequency(df, "date", expected_freq, tolerance=0.3)
    if not valid:
        report["issues"].extend(warnings)

    # 9. Count NaN values
    nan_count = df["value_raw"].isna().sum()
    if nan_count > 0:
        nan_pct = nan_count / len(df) * 100
        report["issues"].append(f"{nan_count} NaN values ({nan_pct:.1f}%)")

    report["final_rows"] = len(df)
    report["nan_count"] = nan_count
    report["date_range"] = {
        "start": df["date"].iloc[0] if len(df) > 0 else None,
        "end": df["date"].iloc[-1] if len(df) > 0 else None,
    }

    return df, report


def compute_transforms(
    df: pd.DataFrame,
    transform: Optional[str] = None,
    value_column: str = "value_raw",
) -> pd.DataFrame:
    """
    Compute derived fields based on transform specification.

    Supported transforms:
        - yoy_pct_change: Year-over-year percentage change
        - mom_pct_change: Month-over-month percentage change
        - mom_change: Month-over-month absolute change
        - log_diff: Log difference (for growth rates)

    Args:
        df: DataFrame with date and value columns
        transform: Transform type from registry
        value_column: Name of value column

    Returns:
        DataFrame with added transformed column
    """
    if df.empty or transform is None or value_column not in df.columns:
        return df

    df = df.copy()

    if transform == "yoy_pct_change":
        # Year-over-year: compare to value 12 months ago
        df["value_transformed"] = df[value_column].pct_change(periods=12) * 100

    elif transform == "mom_pct_change":
        # Month-over-month percentage change
        df["value_transformed"] = df[value_column].pct_change() * 100

    elif transform == "mom_change":
        # Month-over-month absolute change
        df["value_transformed"] = df[value_column].diff()

    elif transform == "log_diff":
        # Log difference (continuous growth rate)
        df["value_transformed"] = np.log(df[value_column]).diff() * 100

    else:
        logger.warning(f"Unknown transform: {transform}")

    return df


def fetch_single_economic_series(
    key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    registry: Optional[Dict[str, Any]] = None,
    apply_transform: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Fetch and normalize data for a single economic series.

    Args:
        key: Series key (e.g., "cpi", "unrate")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        registry: Optional pre-loaded registry
        apply_transform: If True, apply any configured transforms

    Returns:
        Tuple of (normalized DataFrame, fetch report)
    """
    if registry is None:
        registry = load_economic_registry()

    # Find series config
    all_series = registry.get("series", [])
    series_config = None
    for s in all_series:
        if s.get("key", "").lower() == key.lower():
            series_config = s
            break

    if series_config is None:
        return None, {"error": f"Series '{key}' not found in registry"}

    if not series_config.get("enabled", True):
        return None, {"error": f"Series '{key}' is disabled in registry"}

    ticker = series_config.get("ticker", key.upper())
    source = series_config.get("source", "fred")

    report = {
        "key": key,
        "ticker": ticker,
        "source": source,
        "success": False,
        "error": None,
    }

    try:
        # Use existing FRED fetcher
        from prism_engine.fetch.fetcher_fred import FREDFetcher

        fetcher = FREDFetcher()
        df = fetcher.fetch_single(ticker, start_date=start_date, end_date=end_date)

    except ImportError:
        # Fallback: try the 01_fetch path
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "01_fetch"))
            from fetcher_fred import FREDFetcher

            fetcher = FREDFetcher()
            df = fetcher.fetch_single(ticker, start_date=start_date, end_date=end_date)
        except ImportError as e:
            report["error"] = f"Could not import FREDFetcher: {e}"
            logger.error(report["error"])
            return None, report

    except Exception as e:
        report["error"] = f"Fetch failed: {str(e)}"
        logger.error(f"Error fetching {ticker}: {e}")
        return None, report

    if df is None or df.empty:
        report["error"] = "No data returned from source"
        return None, report

    # Normalize the data
    df_clean, norm_report = normalize_economic_dataframe(df, ticker, series_config)
    report.update(norm_report)

    if df_clean.empty:
        report["error"] = "DataFrame empty after normalization"
        return None, report

    # Apply transforms if configured
    if apply_transform:
        transform = series_config.get("transform")
        if transform:
            df_clean = compute_transforms(df_clean, transform, "value_raw")
            report["transform_applied"] = transform

    report["success"] = True
    return df_clean, report


def fetch_all_economic_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    keys: Optional[List[str]] = None,
    write_to_db: bool = False,
    apply_transforms: bool = True,
) -> Dict[str, Any]:
    """
    Fetch economic data for all enabled series in the registry.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        keys: Optional list of specific keys to fetch (None = all enabled)
        write_to_db: If True, write results to database
        apply_transforms: If True, apply configured transforms

    Returns:
        Dictionary with results and summary:
        {
            "success": bool,
            "data": {key: DataFrame, ...},
            "reports": {key: report, ...},
            "summary": {...}
        }
    """
    logger.info("Starting economic data fetch...")

    registry = load_economic_registry()
    enabled = get_enabled_series(registry)

    # Filter to specific keys if provided
    if keys:
        keys_lower = [k.lower() for k in keys]
        enabled = [s for s in enabled if s.get("key", "").lower() in keys_lower]

    results = {
        "success": True,
        "data": {},
        "reports": {},
        "summary": {
            "total_series": len(enabled),
            "successful": 0,
            "failed": 0,
            "total_rows": 0,
        },
    }

    for series in enabled:
        key = series.get("key")
        logger.info(f"Fetching {key}...")

        df, report = fetch_single_economic_series(
            key,
            start_date=start_date,
            end_date=end_date,
            registry=registry,
            apply_transform=apply_transforms,
        )

        results["reports"][key] = report

        if df is not None and not df.empty:
            results["data"][key] = df
            results["summary"]["successful"] += 1
            results["summary"]["total_rows"] += len(df)
            logger.info(f"  -> {len(df)} rows fetched and normalized")
        else:
            results["summary"]["failed"] += 1
            error = report.get("error", "Unknown error")
            logger.warning(f"  -> Failed: {error}")

    # Write to database if requested
    if write_to_db and results["data"]:
        try:
            from data.sql.prism_db import write_dataframe, init_db

            init_db()

            for key, df in results["data"].items():
                # Prepare data for DB
                db_df = df[["date", "value_raw"]].copy()
                db_df.columns = ["date", "value"]

                # Get frequency from registry
                series_config = next(
                    (s for s in enabled if s.get("key") == key),
                    {}
                )
                frequency = series_config.get("frequency", "monthly")
                units = series_config.get("units", "")

                write_dataframe(
                    db_df,
                    indicator_name=key,
                    system="finance",
                    frequency=frequency,
                    source="fred",
                    units=units,
                )
                logger.info(f"  -> Wrote {key} to database")

        except Exception as e:
            logger.error(f"Database write failed: {e}")
            results["summary"]["db_error"] = str(e)

    results["success"] = results["summary"]["failed"] == 0

    logger.info(
        f"Economic fetch complete: {results['summary']['successful']} successful, "
        f"{results['summary']['failed']} failed"
    )

    return results


def validate_economic_data(
    df: pd.DataFrame,
    series_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate economic data against expected characteristics.

    Args:
        df: DataFrame to validate
        series_config: Series configuration from registry

    Returns:
        Validation report dictionary
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": [],
    }

    if df.empty:
        report["valid"] = False
        report["errors"].append("DataFrame is empty")
        return report

    # Check for minimum data points
    expected_freq = series_config.get("frequency", "monthly")
    min_rows = {
        "daily": 100,
        "weekly": 52,
        "monthly": 24,
        "quarterly": 8,
    }.get(expected_freq, 10)

    if len(df) < min_rows:
        report["warnings"].append(
            f"Only {len(df)} rows, expected at least {min_rows} for {expected_freq} data"
        )

    # Check NaN percentage
    if "value_raw" in df.columns:
        nan_pct = df["value_raw"].isna().sum() / len(df)
        if nan_pct > 0.1:
            report["warnings"].append(f"High NaN percentage: {nan_pct:.1%}")
        if nan_pct > 0.5:
            report["valid"] = False
            report["errors"].append(f"Too many NaN values: {nan_pct:.1%}")

    # Check date range
    if "date" in df.columns:
        start = pd.to_datetime(df["date"].iloc[0])
        end = pd.to_datetime(df["date"].iloc[-1])
        span_days = (end - start).days

        if span_days < 365:
            report["warnings"].append(f"Short date range: {span_days} days")

    return report


# Main entry point for CLI usage
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Fetch economic data")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--keys", type=str, nargs="+", help="Specific keys to fetch")
    parser.add_argument("--write-db", action="store_true", help="Write to database")
    parser.add_argument("--no-transform", action="store_true", help="Skip transforms")

    args = parser.parse_args()

    results = fetch_all_economic_data(
        start_date=args.start,
        end_date=args.end,
        keys=args.keys,
        write_to_db=args.write_db,
        apply_transforms=not args.no_transform,
    )

    print("\n" + "=" * 50)
    print("ECONOMIC DATA FETCH SUMMARY")
    print("=" * 50)
    print(f"Total series: {results['summary']['total_series']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Total rows: {results['summary']['total_rows']}")
