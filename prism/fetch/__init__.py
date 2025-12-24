"""
PRISM Fetch System

Data acquisition layer. Fetches raw data from external sources.

Architecture:
    - Fetchers retrieve data and return DataFrames (NO database access)
    - FetchRunner orchestrates fetches and writes to database (ONLY DB writer)
    
Usage:
    from prism.fetch import FetchRunner
    
    runner = FetchRunner()
    summary = runner.run([
        {"id": "GDP", "source": "fred"},
        {"id": "SPY", "source": "tiingo"},
    ])
    summary.print_summary()
"""

from .base import BaseFetcher, FetchResult
from .fred import FredFetcher
from .tiingo import TiingoFetcher
from .runner import FetchRunner

__all__ = [
    "BaseFetcher",
    "FetchResult",
    "FredFetcher",
    "TiingoFetcher",
    "FetchRunner",
]
