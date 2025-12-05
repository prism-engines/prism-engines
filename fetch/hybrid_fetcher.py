import json
from pathlib import Path
import logging

from fetch.fetcher_stooq import StooqFetcher
from fetch.fetcher_yahoo import YahooFetcher

logger = logging.getLogger(__name__)

# Path to fallback JSON map
FALLBACK_MAP_PATH = Path("data/registry/yahoo_fallback_map.json")


def load_fallback_map():
    """Load the Yahoo fallback ticker map."""
    if not FALLBACK_MAP_PATH.exists():
        logger.warning(f"No Yahoo fallback map found: {FALLBACK_MAP_PATH}")
        return {}

    with open(FALLBACK_MAP_PATH) as f:
        return json.load(f)


class HybridFetcher:
    """
    Hybrid Stooq → Yahoo fallback fetcher.
    External JSON config determines Yahoo fallback tickers.
    """

    def __init__(self):
        self.stq = StooqFetcher()
        self.yahoo = YahooFetcher()
        self.fallback_map = load_fallback_map()

    def resolve_yahoo_ticker(self, ticker: str) -> str:
        """Return Yahoo fallback ticker if present."""
        return self.fallback_map.get(ticker.upper(), ticker)

    def fetch_single(self, ticker: str, **kwargs):
        """
        Try Stooq first.
        If empty or invalid → fallback to Yahoo.
        """
        # ---- 1) Try Stooq --------------------------------------
        df_stq = self.stq.fetch_single(ticker, **kwargs)
        if df_stq is not None and len(df_stq) > 0:
            logger.info(f"[HybridFetcher] STQ OK → {ticker}")
            return df_stq

        logger.warning(f"[HybridFetcher] STQ FAIL → {ticker}")

        # ---- 2) Fallback to Yahoo -------------------------------
        yahoo_ticker = self.resolve_yahoo_ticker(ticker)
        logger.info(f"[HybridFetcher] FALLBACK YAHOO → {yahoo_ticker}")

        df_yahoo = self.yahoo.fetch_single(yahoo_ticker, **kwargs)
        if df_yahoo is None or df_yahoo.empty:
            logger.error(f"[HybridFetcher] YAHOO FAIL → {yahoo_ticker}")
            return None

        logger.info(f"[HybridFetcher] YAHOO OK → {yahoo_ticker}")
        return df_yahoo
