#!/usr/bin/env python3
"""
PRISM Fetch Performance Checker
================================

Purpose: Benchmark fetch speed and consistency for data sources.

This script:
- Times FRED fetch for 5 random FRED indicators
- Times Tiingo fetch for 5 random market tickers
- Compares latest date vs today
- Prints warnings for stale sources

Output:
    FETCH PERFORMANCE REPORT
    FRED median fetch time: X ms
    Tiingo median fetch time: X ms
    Stale: [...]

Usage:
    python start/performance/check_fetch_performance.py

Notes:
    - Read-only: never writes to the database
    - Exits gracefully if indicators are missing
"""

import sys
import logging
import random
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from statistics import median

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from panel.runtime_loader import load_panel
from data.sql.db_connector import get_db_path, load_all_indicators_wide
from data.family_manager import FamilyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default indicators for testing
DEFAULT_FRED_INDICATORS = [
    "gdp_growth", "unemployment_rate", "cpi_yoy", "fed_funds_rate",
    "treasury_10y", "industrial_production", "retail_sales", "pce_yoy"
]

DEFAULT_TIINGO_TICKERS = [
    "sp500", "nasdaq", "djia", "vix", "gold_spot",
    "crude_oil", "tnx", "russell_2000"
]

# Staleness threshold
STALE_DAYS = 7


def get_available_indicators_by_source() -> Tuple[List[str], List[str]]:
    """
    Get available indicators grouped by source (FRED vs Tiingo).

    Returns:
        Tuple of (fred_indicators, tiingo_indicators)
    """
    try:
        from data.sql.db_connector import get_connection

        conn = get_connection()
        cursor = conn.cursor()

        # Get indicators with their sources
        cursor.execute("""
            SELECT name, source
            FROM indicators
            WHERE source IN ('fred', 'tiingo')
        """)

        rows = cursor.fetchall()
        conn.close()

        fred_indicators = [row['name'] for row in rows if row['source'] == 'fred']
        tiingo_indicators = [row['name'] for row in rows if row['source'] == 'tiingo']

        return fred_indicators, tiingo_indicators

    except Exception as e:
        logger.warning(f"Could not query indicators by source: {e}")
        # Fall back to defaults
        return DEFAULT_FRED_INDICATORS, DEFAULT_TIINGO_TICKERS


def time_fetch(indicator_names: List[str], source_name: str) -> Dict[str, Any]:
    """
    Time how long it takes to load indicators from the database.

    Args:
        indicator_names: List of indicator names to fetch
        source_name: Name of the source for reporting

    Returns:
        Dict with timing and staleness information
    """
    results = {
        "source": source_name,
        "indicators": [],
        "fetch_times_ms": [],
        "latest_dates": {},
        "stale_indicators": [],
        "missing_indicators": [],
        "errors": []
    }

    today = datetime.now().date()
    stale_threshold = today - timedelta(days=STALE_DAYS)

    for indicator in indicator_names:
        try:
            start_time = time.perf_counter()

            # Load single indicator
            df = load_all_indicators_wide(
                indicators=[indicator],
                start_date=None,
                end_date=None
            )

            end_time = time.perf_counter()
            fetch_time_ms = (end_time - start_time) * 1000

            if df.empty or indicator not in df.columns:
                results["missing_indicators"].append(indicator)
                continue

            results["indicators"].append(indicator)
            results["fetch_times_ms"].append(fetch_time_ms)

            # Check latest date
            if df.index is not None and len(df.index) > 0:
                latest_date = df.index.max()
                if hasattr(latest_date, 'date'):
                    latest_date = latest_date.date()
                elif isinstance(latest_date, str):
                    latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()

                results["latest_dates"][indicator] = latest_date

                if latest_date < stale_threshold:
                    results["stale_indicators"].append({
                        "indicator": indicator,
                        "latest_date": str(latest_date),
                        "days_stale": (today - latest_date).days
                    })

        except Exception as e:
            results["errors"].append({
                "indicator": indicator,
                "error": str(e)
            })
            logger.warning(f"Error fetching {indicator}: {e}")

    return results


def print_report(fred_results: Dict[str, Any], tiingo_results: Dict[str, Any]) -> None:
    """Print the fetch performance report."""
    print("\n" + "=" * 60)
    print("FETCH PERFORMANCE REPORT")
    print("=" * 60)

    # FRED results
    print(f"\n--- FRED Source ({len(fred_results['indicators'])} indicators) ---")
    if fred_results["fetch_times_ms"]:
        median_time = median(fred_results["fetch_times_ms"])
        print(f"FRED median fetch time: {median_time:.2f} ms")
        print(f"FRED min fetch time: {min(fred_results['fetch_times_ms']):.2f} ms")
        print(f"FRED max fetch time: {max(fred_results['fetch_times_ms']):.2f} ms")
    else:
        print("FRED median fetch time: N/A (no indicators fetched)")

    if fred_results["missing_indicators"]:
        print(f"Missing: {fred_results['missing_indicators']}")

    # Tiingo results
    print(f"\n--- Tiingo Source ({len(tiingo_results['indicators'])} indicators) ---")
    if tiingo_results["fetch_times_ms"]:
        median_time = median(tiingo_results["fetch_times_ms"])
        print(f"Tiingo median fetch time: {median_time:.2f} ms")
        print(f"Tiingo min fetch time: {min(tiingo_results['fetch_times_ms']):.2f} ms")
        print(f"Tiingo max fetch time: {max(tiingo_results['fetch_times_ms']):.2f} ms")
    else:
        print("Tiingo median fetch time: N/A (no indicators fetched)")

    if tiingo_results["missing_indicators"]:
        print(f"Missing: {tiingo_results['missing_indicators']}")

    # Staleness warnings
    all_stale = fred_results["stale_indicators"] + tiingo_results["stale_indicators"]
    print(f"\n--- Staleness Check (threshold: {STALE_DAYS} days) ---")
    if all_stale:
        print(f"Stale: {[s['indicator'] for s in all_stale]}")
        for stale in all_stale:
            print(f"  - {stale['indicator']}: last updated {stale['latest_date']} "
                  f"({stale['days_stale']} days ago)")
    else:
        print("No stale indicators detected")

    # Errors
    all_errors = fred_results["errors"] + tiingo_results["errors"]
    if all_errors:
        print(f"\n--- Errors ---")
        for err in all_errors:
            print(f"  - {err['indicator']}: {err['error']}")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for fetch performance check.

    Returns:
        0 on success, 1 on error
    """
    logger.info("Starting fetch performance check...")
    logger.info(f"Database path: {get_db_path()}")

    try:
        # Get available indicators by source
        fred_indicators, tiingo_indicators = get_available_indicators_by_source()

        # Select random sample of 5 each (or all if less than 5)
        sample_size = 5
        fred_sample = random.sample(fred_indicators, min(sample_size, len(fred_indicators))) \
                      if fred_indicators else []
        tiingo_sample = random.sample(tiingo_indicators, min(sample_size, len(tiingo_indicators))) \
                        if tiingo_indicators else []

        logger.info(f"Testing FRED indicators: {fred_sample}")
        logger.info(f"Testing Tiingo indicators: {tiingo_sample}")

        # Run timing tests
        fred_results = time_fetch(fred_sample, "FRED")
        tiingo_results = time_fetch(tiingo_sample, "Tiingo")

        # Print report
        print_report(fred_results, tiingo_results)

        # Return success if no critical errors
        total_errors = len(fred_results["errors"]) + len(tiingo_results["errors"])
        if total_errors > 0:
            logger.warning(f"Completed with {total_errors} errors")
        else:
            logger.info("Fetch performance check completed successfully")

        return 0

    except Exception as e:
        logger.error(f"Fatal error in fetch performance check: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
