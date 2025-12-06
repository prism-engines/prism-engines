#!/usr/bin/env python3
"""
PRISM Unified Fetcher Launcher
==============================

Features:
- Source-based routing (Yahoo vs Stooq)
- Safe length calculations (handles None/empty results)
- Proper result type handling for DataFrame vs dict

Command-line interface for fetching market and economic data.

Usage:
    python start/fetcher.py --all          # Fetch all data
    python start/fetcher.py --market       # Fetch market data only
    python start/fetcher.py --economic     # Fetch economic data only
    python start/fetcher.py --test         # Test connections
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fetch.fetcher_yahoo import YahooFetcher
from fetch.fetcher_stooq import StooqFetcher
from fetch.fetcher_fred import FREDFetcher
from data.registry import load_metric_registry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _safe_len(obj):
    """
    Safely get length of an object.

    Handles:
    - None -> 0
    - Empty DataFrame -> 0
    - DataFrame -> number of columns minus 1 (excluding date)
    - Dict -> len(dict)
    - List -> len(list)
    """
    if obj is None:
        return 0
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return 0
        # For DataFrames, count columns minus the date column
        return max(0, len(obj.columns) - 1)
    if isinstance(obj, (dict, list)):
        return len(obj)
    return 0


def _split_registry_by_source(registry):
    """
    Split market registry items by their data source.

    ALL tickers now route to Stooq (Yahoo is too unreliable).
    Handles ticker format conversion for Stooq.

    Args:
        registry: The loaded metric registry dictionary

    Returns:
        Tuple of (yahoo_registry, stooq_registry) - each a copy with filtered market items
    """
    # Ticker format mapping: Yahoo format -> Stooq format
    # Stooq doesn't use ^ prefix, uses different suffixes
    TICKER_CONVERSIONS = {
        # Indices - Stooq uses different symbols
        "^VIX": "VIX.US",
        "^GSPC": "SPX.US",
        "^NDX": "NDX.US",
        "^DJI": "DJI.US",
        "^RUT": "RUT.US",
        "^FTSE": "FTSE.UK",
        "^GDAXI": "DAX.DE",
        "^FCHI": "CAC.FR",
        "^N225": "NKY.JP",
        "^HSI": "HSI.HK",
        "^STOXX": "SX5E.EU",
        "^AXJO": "ASX.AU",
        "^GSPTSE": "TSX.CA",
        "^BVSP": "BVSP.BR",
        # Dollar Index
        "DX-Y.NYB": "DXY.US",
        # Shanghai - special case
        "000001.SS": "SHCOMP.CN",
    }

    yahoo_items = []  # Empty - we're not using Yahoo anymore
    stooq_items = []

    for item in registry.get("market", []):
        # Get ticker from various locations
        ticker = (
            item.get("params", {}).get("ticker") or
            item.get("ticker") or
            item.get("name", "").upper()
        )

        if not ticker:
            continue

        # Create Stooq item
        stooq_item = dict(item)
        stooq_item["source"] = "stooq"

        # Convert ticker format for Stooq
        if ticker in TICKER_CONVERSIONS:
            stooq_ticker = TICKER_CONVERSIONS[ticker]
        elif ticker.startswith("^"):
            # Generic ^ prefix removal - add .US suffix
            stooq_ticker = ticker[1:] + ".US"
        elif ticker.endswith((".US", ".F", ".UK", ".DE", ".FR", ".JP")):
            # Already in Stooq format
            stooq_ticker = ticker
        else:
            # Regular ticker - add .US suffix
            stooq_ticker = f"{ticker}.US"

        if "params" not in stooq_item:
            stooq_item["params"] = {}
        stooq_item["params"]["ticker"] = stooq_ticker

        stooq_items.append(stooq_item)

    # Create filtered registries
    yahoo_registry = {**registry, "market": yahoo_items}
    stooq_registry = {**registry, "market": stooq_items}

    logger.info(f"Registry: {len(stooq_items)} tickers routed to Stooq (Yahoo disabled)")

    return yahoo_registry, stooq_registry


def fetch_market(registry, start_date=None, end_date=None, write_to_db=True):
    """
    Fetch all enabled market instruments from appropriate sources.

    Routes tickers to Yahoo or Stooq based on the 'source' field in the registry.

    Args:
        registry: The loaded metric registry dictionary
        start_date: Optional start date filter
        end_date: Optional end date filter
        write_to_db: Whether to write to database (not currently used)

    Returns:
        DataFrame of fetched data (may be empty, never None)
    """
    logger.info("=" * 60)
    logger.info("FETCHING MARKET DATA")
    logger.info("=" * 60)

    # Split registry by source
    yahoo_registry, stooq_registry = _split_registry_by_source(registry)

    results = []

    # Fetch from Yahoo
    if yahoo_registry["market"]:
        logger.info("-" * 40)
        logger.info("Fetching from Yahoo Finance...")
        yahoo = YahooFetcher()
        yahoo_df = yahoo.fetch_all(
            registry=yahoo_registry,
            start_date=start_date,
            end_date=end_date
        )
        if yahoo_df is not None and not yahoo_df.empty:
            results.append(yahoo_df)
            logger.info(f"Yahoo: {_safe_len(yahoo_df)} instruments fetched")
        else:
            logger.warning("Yahoo: No data returned")

    # Fetch from Stooq
    if stooq_registry["market"]:
        logger.info("-" * 40)
        logger.info("Fetching from Stooq...")
        stooq = StooqFetcher()
        stooq_df = stooq.fetch_all(
            registry=stooq_registry,
            start_date=start_date,
            end_date=end_date
        )
        if stooq_df is not None and not stooq_df.empty:
            results.append(stooq_df)
            logger.info(f"Stooq: {_safe_len(stooq_df)} instruments fetched")
        else:
            logger.warning("Stooq: No data returned")

    # Merge results
    if not results:
        logger.warning("No market data fetched from any source")
        return pd.DataFrame()

    if len(results) == 1:
        merged = results[0]
    else:
        # Merge all DataFrames on date
        merged = results[0]
        for df in results[1:]:
            merged = pd.merge(merged, df, on="date", how="outer")

    # Sort by date
    if "date" in merged.columns:
        merged = merged.sort_values("date").reset_index(drop=True)

    count = _safe_len(merged)
    logger.info("-" * 40)
    logger.info(f"Market fetch complete: {count} total instruments")
    return merged


def fetch_economic(start_date=None, end_date=None, write_to_db=True):
    """
    Fetch all enabled economic series.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        write_to_db: Whether to write to database

    Returns:
        Dictionary of fetched DataFrames (may be empty, never None)
    """
    logger.info("=" * 60)
    logger.info("FETCHING ECONOMIC DATA")
    logger.info("=" * 60)

    fetcher = FREDFetcher()
    results = fetcher.fetch_all(
        write_to_db=write_to_db,
        start_date=start_date,
        end_date=end_date
    )

    # Safe count calculation
    count = _safe_len(results)
    logger.info(f"Economic fetch complete: {count} series")
    return results if results is not None else {}


def test_connections():
    """Test all data source connections."""
    logger.info("=" * 60)
    logger.info("TESTING CONNECTIONS")
    logger.info("=" * 60)

    results = {}

    # Test Yahoo
    logger.info("Testing Yahoo Finance...")
    yahoo = YahooFetcher()
    try:
        test_df = yahoo.fetch_single("SPY", start_date="2024-01-01", end_date="2024-01-05")
        results["yahoo"] = test_df is not None and not test_df.empty
    except Exception as e:
        logger.error(f"Yahoo test error: {e}")
        results["yahoo"] = False
    logger.info(f"  Yahoo Finance: {'OK' if results['yahoo'] else 'FAILED'}")

    # Test Stooq
    logger.info("Testing Stooq...")
    stooq = StooqFetcher()
    try:
        test_df = stooq.fetch_single("SPY.US", start_date="2024-01-01", end_date="2024-01-05")
        results["stooq"] = test_df is not None and not test_df.empty
    except Exception as e:
        logger.error(f"Stooq test error: {e}")
        results["stooq"] = False
    logger.info(f"  Stooq: {'OK' if results['stooq'] else 'FAILED'}")

    # Test FRED
    logger.info("Testing FRED API...")
    fred = FREDFetcher()
    results["fred"] = fred.test_connection()
    logger.info(f"  FRED API: {'OK' if results['fred'] else 'FAILED'}")

    # Summary
    all_ok = all(results.values())
    logger.info("=" * 60)
    logger.info(f"Connection test: {'ALL PASSED' if all_ok else 'SOME FAILED'}")

    return results


def show_database_status():
    """Show database status and statistics."""
    logger.info("=" * 60)
    logger.info("DATABASE STATUS")
    logger.info("=" * 60)

    try:
        from data.sql.db import get_db_path, get_table_stats, list_indicators

        db_path = get_db_path()
        logger.info(f"Database path: {db_path}")

        stats = get_table_stats()
        logger.info("\nTable statistics:")
        for table, count in stats.items():
            logger.info(f"  {table}: {count} rows")

        # Show indicators by system
        indicators = list_indicators()
        systems = {}
        for ind in indicators:
            sys = ind["system"]
            systems[sys] = systems.get(sys, 0) + 1

        if systems:
            logger.info("\nIndicators by system:")
            for sys, count in sorted(systems.items()):
                logger.info(f"  {sys}: {count} indicators")

    except Exception as e:
        logger.error(f"Could not get database status: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Unified Data Fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start/fetcher.py --all
    python start/fetcher.py --market --start-date 2020-01-01
    python start/fetcher.py --economic --no-db
    python start/fetcher.py --test
    python start/fetcher.py --status
        """
    )

    # Data type selection
    parser.add_argument(
        "--market", "-m",
        action="store_true",
        help="Fetch market data (Yahoo + Stooq)"
    )
    parser.add_argument(
        "--economic", "-e",
        action="store_true",
        help="Fetch economic data (FRED)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Fetch all data types"
    )

    # Date filters
    parser.add_argument(
        "--start-date", "-s",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", "-E",
        type=str,
        help="End date (YYYY-MM-DD)"
    )

    # Options
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Don't write to database (fetch only)"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test connections only"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show database status"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle commands
    if args.test:
        results = test_connections()
        return 0 if all(results.values()) else 1

    if args.status:
        show_database_status()
        return 0

    # If no data type specified, show help
    if not (args.all or args.market or args.economic):
        parser.print_help()
        return 0

    # Fetch data
    write_to_db = not args.no_db
    results = {"market": None, "economic": {}}

    # Load the registry (required by unified fetcher interface)
    try:
        registry = load_metric_registry()
    except Exception as e:
        logger.error(f"Failed to load registry: {e}")
        logger.error("Make sure PyYAML is installed: pip install pyyaml")
        return 1

    if args.all or args.market:
        results["market"] = fetch_market(
            registry=registry,
            start_date=args.start_date,
            end_date=args.end_date,
            write_to_db=write_to_db
        )

    if args.all or args.economic:
        results["economic"] = fetch_economic(
            start_date=args.start_date,
            end_date=args.end_date,
            write_to_db=write_to_db
        )

    # Summary - with safe length calculations
    logger.info("=" * 60)
    logger.info("FETCH SUMMARY")
    logger.info("=" * 60)

    market_count = _safe_len(results.get('market'))
    econ_count = _safe_len(results.get('economic'))

    logger.info(f"Market instruments: {market_count}")
    logger.info(f"Economic series: {econ_count}")

    if write_to_db:
        show_database_status()

    return 0


if __name__ == "__main__":
    sys.exit(main())
