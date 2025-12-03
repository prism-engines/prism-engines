"""
PRISM Engine - Stage 01: Data Fetching
Fetches data from various sources (FRED, Yahoo Finance, Climate APIs)
"""

from .fetcher_base import BaseFetcher
from .fetcher_fred import FREDFetcher
from .fetcher_yahoo import YahooFetcher
from .fetcher_climate import ClimateFetcher
from .fetcher_custom import CustomFetcher
from .fetcher_stooq import StooqFetcher, fetch_stooq, STOOQ_TICKER_MAP

__all__ = [
    'BaseFetcher',
    'FREDFetcher',
    'YahooFetcher',
    'ClimateFetcher',
    'CustomFetcher',
    'StooqFetcher',
    'fetch_stooq',
    'STOOQ_TICKER_MAP',
    'fetch_market',
]


def fetch_market(ticker: str, start: str = "2000-01-01", end: str = None, source: str = "auto") -> dict:
    """Fetch market data with automatic fallback: Stooq -> Yahoo."""
    sources = ["stooq", "yahoo"] if source == "auto" else [source]

    for src in sources:
        try:
            if src == "stooq":
                df = fetch_stooq(ticker, start, end)
            elif src == "yahoo":
                from .fetcher_yahoo import YahooFetcher
                fetcher = YahooFetcher()
                df = fetcher.fetch(ticker, start, end)
            else:
                raise ValueError(f"Unknown source: {src}")

            if df is not None and len(df) > 0:
                return {"data": df, "source": src}
        except:
            continue

    raise ValueError(f"All sources failed for {ticker}")
