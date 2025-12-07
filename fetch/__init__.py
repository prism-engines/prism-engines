"""
PRISM Engine - Data Fetching Module

Fetchers for various data sources:
- PRISMFetcher: Unified fetcher (FRED for economy, Tiingo for market)
- StooqFetcher: Primary market data (US ETFs, stocks, commodities)
- TiingoFetcher: Reliable backup + crypto data
- FREDFetcher: Economic data from FRED

Usage:
    from fetch import PRISMFetcher, TiingoFetcher, FREDFetcher

    # Unified fetcher (recommended)
    fetcher = PRISMFetcher()
    economy_df = fetcher.fetch_panel('economy')  # Uses FRED
    market_df = fetcher.fetch_panel('market')    # Uses Tiingo

    # Individual fetchers
    tiingo = TiingoFetcher(api_key="your_key")
    df = tiingo.fetch_single("SPY")
    df = tiingo.fetch_crypto("btcusd")
"""

from .fetcher_base import BaseFetcher
from .fetcher_fred import FREDFetcher, FetcherFRED
from .fetcher_stooq import StooqFetcher
from .fetcher_tiingo import TiingoFetcher, FetcherTiingo
from .fetcher_unified import PRISMFetcher, fetch_prism_data

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
    'PRISMFetcher',
    'fetch_prism_data',
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
