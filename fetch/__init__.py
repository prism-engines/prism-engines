"""
PRISM Engine - Data Fetching Module

Fetchers for various data sources:
- YahooFetcher: Market data from Yahoo Finance
- FREDFetcher: Economic data from FRED
- ClimateFetcher: Climate data (placeholder)
- CustomFetcher: Custom data sources

Usage:
    from fetch import YahooFetcher, FREDFetcher
    
    yahoo = YahooFetcher()
    fred = FREDFetcher()
    
    # Fetch single
    df = yahoo.fetch_single("SPY")
    
    # Fetch all from registry
    results = yahoo.fetch_all()
"""

from .fetcher_base import BaseFetcher
from .fetcher_fred import FREDFetcher, FetcherFRED
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
    'FetcherFRED',  # alias
    'YahooFetcher',
    'FetcherYahoo',  # alias
    'ClimateFetcher',
    'CustomFetcher',
]
