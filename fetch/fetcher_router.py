"""
PRISM Hybrid Fetcher Router
---------------------------
Primary: Stooq
Fallback: Yahoo (with automatic ticker remapping)
"""

import logging
from fetch.fetcher_stooq import StooqFetcher
from fetch.fetcher_yahoo import YahooFetcher

logger = logging.getLogger(__name__)


# -----------------------------------------
# FIX: special Yahoo symbols mapping
# -----------------------------------------
YAHOO_REMAP = {
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
}


def map_to_yahoo(ticker: str) -> str:
    """
    Convert Stooq tickers (SPY.US, VIX.US, etc.) into Yahoo equivalents.
    """
    base = ticker.replace(".US", "").lower()

    # special symbols:
    if base in YAHOO_REMAP:
        return YAHOO_REMAP[base]

    # simple case: SPY.US → SPY
    return base.upper()


class HybridFetcher:
    """
    Hybrid fetcher:
    - Try Stooq first
    - If Stooq fails or is empty → Yahoo fallback
    """

    def __init__(self):
        self.stooq = StooqFetcher()
        self.yahoo = YahooFetcher()

    def fetch_single(self, ticker: str, **kwargs):
        logger.info(f"[HybridFetcher] Attempt STQ → {ticker}")

        # ---- STQOOQ ATTEMPT ----
        try:
            df = self.stooq.fetch_single(ticker, **kwargs)
            if df is not None and len(df) > 0:
                logger.info(f"[HybridFetcher] STQ OK → {ticker} ({len(df)} rows)")
                return df
        except Exception as e:
            logger.warning(f"[HybridFetcher] STQ FAIL → {ticker}: {e}")

        # --------------------------------------
        # FALLBACK: YAHOO
        # --------------------------------------

        yahoo_ticker = map_to_yahoo(ticker)
        logger.info(f"[HybridFetcher] FALLBACK YAHOO → {yahoo_ticker}")

        try:
            df = self.yahoo.fetch_single(yahoo_ticker, **kwargs)
            if df is not None and len(df) > 0:
                logger.info(f"[HybridFetcher] YAHOO OK → {yahoo_ticker} ({len(df)} rows)")
                return df
            else:
                logger.error(f"[HybridFetcher] Yahoo returned empty for {yahoo_ticker}")
        except Exception as e:
            logger.error(f"[HybridFetcher] Yahoo FAILURE for {yahoo_ticker}: {e}")

        # total failure
        logger.error(f"[HybridFetcher] TOTAL FAILURE → {ticker}")
        return None