"""
PRISM Engine - Data Fetching Module

Fetchers for various data sources:
- StooqFetcher: Primary market data (US ETFs, stocks, commodities)
- TiingoFetcher: Reliable backup + crypto data
- FREDFetcher: Economic data from FRED
- YahooFetcher: Legacy (deprecated - use TiingoFetcher)

Usage:
    from fetch import StooqFetcher, TiingoFetcher, FREDFetcher

    stooq = StooqFetcher()
    tiingo = TiingoFetcher(api_key="your_key")
    fred = FREDFetcher()

    # Fetch single
    df = stooq.fetch_single("SPY.US")
    df = tiingo.fetch_single("SPY")
    df = tiingo.fetch_crypto("btcusd")
"""

from .fetcher_base import BaseFetcher
from .fetcher_fred import FREDFetcher, FetcherFRED
from .fetcher_stooq import StooqFetcher
from .fetcher_tiingo import TiingoFetcher, FetcherTiingo

# Legacy Yahoo fetcher (deprecated)
from .fetcher_yahoo import YahooFetcher, FetcherYahoo

# Optional fetchers (may not be fully implemented)
try:
    from .fetcher_climate import ClimateFetcher
except ImportError:
    ClimateFetcher = None

try:
    from .fetcher_custom import CustomFetcher
except ImportError:
    CustomFetcher = None

__all__ = [
    'BaseFetcher',
    'FREDFetcher',
    'FetcherFRED',
    'StooqFetcher',
    'TiingoFetcher',
    'FetcherTiingo',
    'YahooFetcher',
    'FetcherYahoo',
    'ClimateFetcher',
    'CustomFetcher',
]
