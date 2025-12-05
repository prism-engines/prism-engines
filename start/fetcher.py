#!/usr/bin/env python3
"""
PRISM Unified Fetcher Launcher
==============================

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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fetch.fetcher_yahoo import YahooFetcher
from fetch.fetcher_fred import FREDFetcher
from data.registry import load_metric_registry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_market(registry, start_date=None, end_date=None, write_to_db=True):
    """
    Fetch all enabled market instruments.

    Args:
        registry: The loaded metric registry dictionary
        start_date: Optional start date filter
        end_date: Optional end date filter
        write_to_db: Whether to write to database

    Returns:
        Dictionary of fetched DataFrames
    """
    logger.info("=" * 60)
    logger.info("FETCHING MARKET DATA")
    logger.info("=" * 60)

    fetcher = YahooFetcher()
    # Pass registry to fetch_all (required by new unified fetcher interface)
    results = fetcher.fetch_all(
        registry=registry,
        start_date=start_date,
        end_date=end_date,
        write_to_db=write_to_db
    )
    
    logger.info(f"Market fetch complete: {len(results)} instruments")
    return results


def fetch_economic(start_date=None, end_date=None, write_to_db=True):
    """
    Fetch all enabled economic series.
    
    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        write_to_db: Whether to write to database
        
    Returns:
        Dictionary of fetched DataFrames
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
    
    logger.info(f"Economic fetch complete: {len(results)} series")
    return results


def test_connections():
    """Test all data source connections."""
    logger.info("=" * 60)
    logger.info("TESTING CONNECTIONS")
    logger.info("=" * 60)
    
    results = {}
    
    # Test Yahoo
    logger.info("Testing Yahoo Finance...")
    yahoo = YahooFetcher()
    results["yahoo"] = yahoo.test_connection()
    logger.info(f"  Yahoo Finance: {'OK' if results['yahoo'] else 'FAILED'}")
    
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
        help="Fetch market data (Yahoo Finance)"
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
    results = {"market": {}, "economic": {}}

    # Load the registry (required by unified fetcher interface)
    registry = load_metric_registry()

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
    
    # Summary
    logger.info("=" * 60)
    logger.info("FETCH SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Market instruments: {len(results['market'])}")
    logger.info(f"Economic series: {len(results['economic'])}")
    
    if write_to_db:
        show_database_status()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
