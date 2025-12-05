"""
Custom Fetcher Routing
----------------------
Maps source names → fetcher classes.

This allows any registry entry to specify:
    "source": "stooq"
    "source": "yahoo"
    "source": "fred"
"""

from fetch.fetcher_stooq import StooqFetcher
from fetch.fetcher_yahoo import YahooFetcher
from fetch.fetcher_fred import FREDFetcher


FETCHER_MAP = {
    "stooq": StooqFetcher,   # PRIMARY
    "yahoo": YahooFetcher,   # Exceptions only (VIX, DXY)
    "fred":  FREDFetcher,    # Economic / Treasury yields
}


def get_fetcher(source_name: str):
    """
    Return instantiated fetcher for specified source.
    """
    source_name = source_name.lower()

    if source_name not in FETCHER_MAP:
        raise ValueError(f"Unknown data source: {source_name}")

    return FETCHER_MAP[source_name]()