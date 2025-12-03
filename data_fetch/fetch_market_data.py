"""
Market Data Fetcher
===================

Registry-driven market data fetching with strict normalization and validation.

This module fetches market data (equities, bonds, commodities, currencies)
based on the market_registry.json configuration.

Key Features:
    - Registry-driven: All tickers defined in market_registry.json
    - Strict normalization: ISO dates, float prices, no garbage
    - Validation: No duplicates, no future dates, sorted ascending
    - Database integration: Passes clean DataFrames to DB writer

Usage:
    from data_fetch.fetch_market_data import fetch_all_market_data

    # Fetch all enabled instruments
    results = fetch_all_market_data()

    # Fetch single instrument
    df = fetch_single_market_instrument("spy")
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Import existing fetchers
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
MARKET_REGISTRY_PATH = REGISTRY_DIR / "market_registry.json"
SYSTEM_REGISTRY_PATH = REGISTRY_DIR / "system_registry.json"


class MarketFetchError(Exception):
    """Raised when market data fetch fails."""
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


def load_market_registry() -> Dict[str, Any]:
    """
    Load the market registry configuration.

    Returns:
        Dictionary with market instruments configuration
    """
    if not MARKET_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Market registry not found: {MARKET_REGISTRY_PATH}")

    with open(MARKET_REGISTRY_PATH, "r") as f:
        return json.load(f)


def get_enabled_instruments(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get list of enabled instruments from registry.

    Args:
        registry: Market registry dictionary

    Returns:
        List of enabled instrument configurations
    """
    instruments = registry.get("instruments", [])
    return [inst for inst in instruments if inst.get("enabled", True)]


def normalize_market_dataframe(
    df: pd.DataFrame,
    ticker: str,
    instrument_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize market data DataFrame to standard format.

    Normalization includes:
        - Date column: ISO format YYYY-MM-DD
        - Price column: float values
        - Dividend column: float values (if present)
        - Remove duplicates, sort ascending, remove garbage

    Args:
        df: Raw DataFrame from fetcher
        ticker: Ticker symbol
        instrument_config: Instrument configuration from registry

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
    price_col = None
    volume_col = None

    # Find the price column (usually named after ticker or 'close')
    ticker_lower = ticker.lower()
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == ticker_lower or col_lower == "close" or col_lower == f"{ticker_lower}_close":
            price_col = col
            break
        if "close" in col_lower:
            price_col = col
            break

    if price_col is None:
        # Try to find any numeric column that could be price
        for col in df.columns:
            if col.lower() != "date" and "volume" not in col.lower():
                price_col = col
                break

    if price_col is None:
        report["issues"].append("Could not identify price column")
        return pd.DataFrame(), report

    # Find volume column if present
    for col in df.columns:
        if "volume" in col.lower():
            volume_col = col
            break

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
    value_cols = [price_col]
    if volume_col:
        value_cols.append(volume_col)

    df, n_garbage = remove_footer_garbage(df, date_col, value_cols, log_removed=False)
    if n_garbage > 0:
        report["fixes"].append(f"Removed {n_garbage} footer garbage rows")

    # 3. Clean numeric columns
    df = clean_numeric_column(df, price_col, drop_invalid=True)
    if volume_col and volume_col in df.columns:
        df = clean_numeric_column(df, volume_col, drop_invalid=False)

    # 4. Remove duplicate dates
    df, n_dups = remove_duplicate_dates(df, date_col, keep="last", log_removed=False)
    if n_dups > 0:
        report["fixes"].append(f"Removed {n_dups} duplicate dates")

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
    rename_map = {
        date_col: "date",
        price_col: "price",
    }
    if volume_col and volume_col in df.columns:
        rename_map[volume_col] = "volume"

    df = df.rename(columns=rename_map)

    # Select final columns
    final_cols = ["date", "price"]
    if "volume" in df.columns:
        final_cols.append("volume")

    # Add dividend column if available (set to 0 if not)
    if "dividend" not in df.columns:
        df["dividend"] = 0.0
    final_cols.append("dividend")

    df = df[final_cols]

    # 8. Validate frequency
    expected_freq = instrument_config.get("frequency", "daily")
    valid, warnings = validate_frequency(df, "date", expected_freq)
    if not valid:
        report["issues"].extend(warnings)

    report["final_rows"] = len(df)
    report["date_range"] = {
        "start": df["date"].iloc[0] if len(df) > 0 else None,
        "end": df["date"].iloc[-1] if len(df) > 0 else None,
    }

    return df, report


def fetch_single_market_instrument(
    key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    registry: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Fetch and normalize data for a single market instrument.

    Args:
        key: Instrument key (e.g., "spy", "tlt")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        registry: Optional pre-loaded registry

    Returns:
        Tuple of (normalized DataFrame, fetch report)
    """
    if registry is None:
        registry = load_market_registry()

    # Find instrument config
    instruments = registry.get("instruments", [])
    instrument_config = None
    for inst in instruments:
        if inst.get("key", "").lower() == key.lower():
            instrument_config = inst
            break

    if instrument_config is None:
        return None, {"error": f"Instrument '{key}' not found in registry"}

    if not instrument_config.get("enabled", True):
        return None, {"error": f"Instrument '{key}' is disabled in registry"}

    ticker = instrument_config.get("ticker", key.upper())
    source = instrument_config.get("source", "yahoo")

    report = {
        "key": key,
        "ticker": ticker,
        "source": source,
        "success": False,
        "error": None,
    }

    try:
        # Use existing Yahoo fetcher
        from prism_engine.fetch.fetcher_yahoo import YahooFetcher

        fetcher = YahooFetcher()
        df = fetcher.fetch_single(ticker, start_date=start_date, end_date=end_date)

    except ImportError:
        # Fallback: try the 01_fetch path
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "01_fetch"))
            from fetcher_yahoo import YahooFetcher

            fetcher = YahooFetcher()
            df = fetcher.fetch_single(ticker, start_date=start_date, end_date=end_date)
        except ImportError as e:
            report["error"] = f"Could not import YahooFetcher: {e}"
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
    df_clean, norm_report = normalize_market_dataframe(df, ticker, instrument_config)
    report.update(norm_report)

    if df_clean.empty:
        report["error"] = "DataFrame empty after normalization"
        return None, report

    report["success"] = True
    return df_clean, report


def fetch_all_market_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    keys: Optional[List[str]] = None,
    write_to_db: bool = False,
) -> Dict[str, Any]:
    """
    Fetch market data for all enabled instruments in the registry.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        keys: Optional list of specific keys to fetch (None = all enabled)
        write_to_db: If True, write results to database

    Returns:
        Dictionary with results and summary:
        {
            "success": bool,
            "data": {key: DataFrame, ...},
            "reports": {key: report, ...},
            "summary": {...}
        }
    """
    logger.info("Starting market data fetch...")

    registry = load_market_registry()
    enabled = get_enabled_instruments(registry)

    # Filter to specific keys if provided
    if keys:
        keys_lower = [k.lower() for k in keys]
        enabled = [i for i in enabled if i.get("key", "").lower() in keys_lower]

    results = {
        "success": True,
        "data": {},
        "reports": {},
        "summary": {
            "total_instruments": len(enabled),
            "successful": 0,
            "failed": 0,
            "total_rows": 0,
        },
    }

    for inst in enabled:
        key = inst.get("key")
        logger.info(f"Fetching {key}...")

        df, report = fetch_single_market_instrument(
            key,
            start_date=start_date,
            end_date=end_date,
            registry=registry,
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
                db_df = df[["date", "price"]].copy()
                db_df.columns = ["date", "value"]

                write_dataframe(
                    db_df,
                    indicator_name=key,
                    system="finance",
                    frequency="daily",
                    source="yahoo",
                )
                logger.info(f"  -> Wrote {key} to database")

        except Exception as e:
            logger.error(f"Database write failed: {e}")
            results["summary"]["db_error"] = str(e)

    results["success"] = results["summary"]["failed"] == 0

    logger.info(
        f"Market fetch complete: {results['summary']['successful']} successful, "
        f"{results['summary']['failed']} failed"
    )

    return results


def compute_daily_returns(
    df: pd.DataFrame,
    price_column: str = "price",
) -> pd.DataFrame:
    """
    Compute daily price returns.

    Args:
        df: DataFrame with date and price columns
        price_column: Name of price column

    Returns:
        DataFrame with added 'return' column
    """
    if df.empty or price_column not in df.columns:
        return df

    df = df.copy()
    df["return"] = df[price_column].pct_change()

    return df


# Main entry point for CLI usage
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Fetch market data")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--keys", type=str, nargs="+", help="Specific keys to fetch")
    parser.add_argument("--write-db", action="store_true", help="Write to database")

    args = parser.parse_args()

    results = fetch_all_market_data(
        start_date=args.start,
        end_date=args.end,
        keys=args.keys,
        write_to_db=args.write_db,
    )

    print("\n" + "=" * 50)
    print("MARKET DATA FETCH SUMMARY")
    print("=" * 50)
    print(f"Total instruments: {results['summary']['total_instruments']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Total rows: {results['summary']['total_rows']}")
