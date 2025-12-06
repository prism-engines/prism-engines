"""
PRISM Engine - Data Fetching Module

Fetchers for various data sources:
- StooqFetcher: Primary market data (Stooq daily CSV)
- YahooFetcher: Market data from Yahoo Finance (fallback / special cases)
- FREDFetcher: Economic data from FRED
- HybridFetcher: Stooq primary + Yahoo fallback wrapper
- ClimateFetcher: Climate data (optional / placeholder)
- CustomFetcher: Custom data sources

Usage examples:

    from fetch import StooqFetcher, YahooFetcher, FREDFetcher, HybridFetcher

    stq = StooqFetcher()
    yfh = YahooFetcher()
    fred = FREDFetcher()
    hybrid = HybridFetcher()
"""

from .fetcher_base import BaseFetcher
from .fetcher_fred import FREDFetcher
from .fetcher_yahoo import YahooFetcher
from .fetcher_stooq import StooqFetcher
from .hybrid_fetcher import HybridFetcher

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
    "BaseFetcher",
    "FREDFetcher",
    "YahooFetcher",
    "StooqFetcher",
    "HybridFetcher",
    "ClimateFetcher",
    "CustomFetcher",
]