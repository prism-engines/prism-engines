#!/usr/bin/env python3
"""
PRISM Fetch Script

Command-line entry point for fetching indicator data.

Usage:
    # Fetch all indicators in registry
    python scripts/fetch.py
    
    # Fetch specific indicators
    python scripts/fetch.py --indicators GDP UNRATE SPY
    
    # Fetch by source
    python scripts/fetch.py --source fred
    
    # Fetch by category
    python scripts/fetch.py --category rates
    
    # Fetch a predefined panel
    python scripts/fetch.py --panel pilot
"""

import argparse
import sys
from pathlib import Path
from datetime import date

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prism.fetch import FetchRunner
from prism.registry import RegistryLoader
from prism.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch indicator data for PRISM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/fetch.py                      # Fetch all
    python scripts/fetch.py --indicators GDP SPY # Fetch specific
    python scripts/fetch.py --source fred        # Fetch by source
    python scripts/fetch.py --panel pilot        # Fetch panel
        """
    )
    
    # Selection options (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--indicators", "-i",
        nargs="+",
        help="Specific indicator IDs to fetch"
    )
    group.add_argument(
        "--source", "-s",
        choices=["fred", "tiingo"],
        help="Fetch all indicators from a source"
    )
    group.add_argument(
        "--category", "-c",
        help="Fetch all indicators in a category"
    )
    group.add_argument(
        "--panel", "-p",
        help="Fetch a predefined panel of indicators"
    )
    
    # Date range
    parser.add_argument(
        "--start-date",
        type=lambda s: date.fromisoformat(s),
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: date.fromisoformat(s),
        help="End date (YYYY-MM-DD)"
    )
    
    # Other options
    parser.add_argument(
        "--registry",
        type=Path,
        help="Path to custom registry YAML"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)
    
    # Load registry
    registry = RegistryLoader(args.registry)
    
    # Determine which indicators to fetch
    if args.indicators:
        indicators = registry.get_indicators(args.indicators)
        if not indicators:
            print(f"ERROR: No matching indicators found for: {args.indicators}")
            sys.exit(1)
    elif args.source:
        indicators = registry.get_by_source(args.source)
        if not indicators:
            print(f"ERROR: No indicators found for source: {args.source}")
            sys.exit(1)
    elif args.category:
        indicators = registry.get_by_category(args.category)
        if not indicators:
            print(f"ERROR: No indicators found for category: {args.category}")
            sys.exit(1)
    elif args.panel:
        # Load panel from registry YAML
        # For now, use get_indicators with panel names
        # TODO: Add panel support to RegistryLoader
        print(f"ERROR: Panel support not yet implemented")
        sys.exit(1)
    else:
        # Default: fetch all
        indicators = registry.get_all()
    
    if not indicators:
        print("ERROR: No indicators to fetch")
        sys.exit(1)
    
    print(f"Fetching {len(indicators)} indicators...")
    
    # Convert to fetch list format
    fetch_list = registry.to_fetch_list(indicators)
    
    # Run fetch
    runner = FetchRunner()
    
    try:
        summary = runner.run(
            indicators=fetch_list,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        summary.print_summary()
        
        # Exit with error code if any failed
        if summary.failed > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
