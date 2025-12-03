"""
PRISM Fetcher Launcher
Uses modern fetchers from fetch/ directory
"""

import argparse
import logging
from fetch.fetcher_yahoo import YahooFetcher
from fetch.fetcher_fred import FREDFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_market():
    logger.info("=== Fetching MARKET data ===")
    yf = YahooFetcher()
    yf.fetch_all()


def fetch_economic():
    logger.info("=== Fetching ECONOMIC data ===")
    ff = FREDFetcher()
    ff.fetch_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM Unified Fetcher")
    parser.add_argument("--market", action="store_true", help="Fetch market data")
    parser.add_argument("--economic", action="store_true", help="Fetch economic data")
    parser.add_argument("--all", action="store_true", help="Fetch all datasets")

    args = parser.parse_args()

    if args.all or args.market:
        fetch_market()

    if args.all or args.economic:
        fetch_economic()
