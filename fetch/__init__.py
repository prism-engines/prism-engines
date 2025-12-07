"""
PRISM Engine - Data Fetching Module (Registry v2)
==================================================

Source-based fetchers for PRISM's unified indicator system:

- SourceRouter: Registry-aware routing to FRED or Tiingo
- FREDFetcher: Economic data from Federal Reserve (FRED)
- TiingoFetcher: ETF/Stock adjusted close prices

API Keys (via environment variables):
- FRED_API_KEY: Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
- TIINGO_API_KEY: Get free key at https://api.tiingo.com

Usage:
    from fetch import SourceRouter, FREDFetcher, TiingoFetcher

    # Recommended: Use SourceRouter for registry-based fetching
    router = SourceRouter()
    df = router.fetch("sp500")   # Routes to FRED based on registry
    df = router.fetch("spy")     # Routes to Tiingo based on registry

    # Direct fetcher usage
    fred = FREDFetcher()
    df = fred.fetch_single("DGS10")

    tiingo = TiingoFetcher()
    df = tiingo.fetch_single("SPY")
"""

from .fetcher_base import BaseFetcher
from .fetcher_fred import FREDFetcher, FetcherFRED
from .fetcher_tiingo import TiingoFetcher, FetcherTiingo
from .fetcher_router import SourceRouter, HybridFetcher

# Legacy unified fetcher (may still be used by some code)
try:
    from .fetcher_unified import PRISMFetcher, fetch_prism_data
except ImportError:
    PRISMFetcher = None
    fetch_prism_data = None

# Optional fetchers (may not be fully implemented)
try:
    from .fetcher_climate import ClimateFetcher
except ImportError:
    ClimateFetcher = None

try:
    from .fetcher_custom import CustomFetcher
except ImportError:
    CustomFetcher = None

# Deprecated: Yahoo and Stooq fetchers
# These are kept for backward compatibility but should not be used
try:
    from .fetcher_yahoo import YahooFetcher, FetcherYahoo
except ImportError:
    YahooFetcher = None
    FetcherYahoo = None

try:
    from .fetcher_stooq import StooqFetcher
except ImportError:
    StooqFetcher = None

__all__ = [
    # Core
    'BaseFetcher',
    'SourceRouter',
    # Primary fetchers
    'FREDFetcher',
    'FetcherFRED',
    'TiingoFetcher',
    'FetcherTiingo',
    # Legacy (deprecated)
    'HybridFetcher',  # Alias for SourceRouter
    'PRISMFetcher',
    'fetch_prism_data',
    'YahooFetcher',
    'FetcherYahoo',
    'StooqFetcher',
    # Optional
    'ClimateFetcher',
    'CustomFetcher',
]
