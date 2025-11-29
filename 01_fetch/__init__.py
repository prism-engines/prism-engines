"""
PRISM Engine - Stage 01: Data Fetching
Fetches data from various sources (FRED, Yahoo Finance, Climate APIs)
"""

from .fetcher_base import BaseFetcher
from .fetcher_fred import FREDFetcher
from .fetcher_yahoo import YahooFetcher
from .fetcher_climate import ClimateFetcher
from .fetcher_custom import CustomFetcher

__all__ = [
    'BaseFetcher',
    'FREDFetcher',
    'YahooFetcher',
    'ClimateFetcher',
    'CustomFetcher',
]
