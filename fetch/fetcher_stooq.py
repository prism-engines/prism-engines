"""
Stooq Fetcher - Primary market data source for PRISM

Stooq provides stable, clean CSV data for major indices and ETFs.
No API key required, no MultiIndex issues.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)

# Ticker mapping: Yahoo format -> Stooq format
STOOQ_TICKER_MAP = {
    # Major indices
    "SPY": "SPY.US", "QQQ": "QQQ.US", "IWM": "IWM.US", "DIA": "DIA.US",
    # Sector ETFs
    "XLK": "XLK.US", "XLF": "XLF.US", "XLE": "XLE.US", "XLV": "XLV.US",
    "XLI": "XLI.US", "XLP": "XLP.US", "XLY": "XLY.US", "XLU": "XLU.US",
    "XLB": "XLB.US", "XLRE": "XLRE.US", "XLC": "XLC.US",
    # Bond ETFs
    "TLT": "TLT.US", "IEF": "IEF.US", "SHY": "SHY.US", "BND": "BND.US",
    "LQD": "LQD.US", "HYG": "HYG.US", "TIP": "TIP.US",
    # Commodity ETFs
    "GLD": "GLD.US", "SLV": "SLV.US", "USO": "USO.US",
    # Volatility & Currency
    "^VIX": "VIX.US", "VIX": "VIX.US", "DXY": "DXY.US", "^DXY": "DXY.US",
}


class StooqFetcher(BaseFetcher):
    """Fetcher for Stooq market data."""

    SOURCE_NAME = "stooq"
    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.timeout = self.config.get("timeout", 30)

    def _to_stooq_ticker(self, ticker: str) -> str:
        """Convert Yahoo-style ticker to Stooq format."""
        if ticker.upper() in STOOQ_TICKER_MAP:
            return STOOQ_TICKER_MAP[ticker.upper()]
        ticker_upper = ticker.upper()
        return f"{ticker_upper}.US" if not ticker_upper.endswith(".US") else ticker_upper

    def fetch(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical data from Stooq."""
        stooq_ticker = self._to_stooq_ticker(ticker)
        url = f"{self.BASE_URL}?s={stooq_ticker}&i=d"

        logger.info(f"Fetching {ticker} ({stooq_ticker}) from Stooq")

        try:
            df = pd.read_csv(url, timeout=self.timeout)
            if df.empty:
                return pd.DataFrame()

            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"])

            if start:
                df = df[df["date"] >= pd.to_datetime(start)]
            if end:
                df = df[df["date"] <= pd.to_datetime(end)]

            df = df.sort_values("date").reset_index(drop=True)
            cols = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]
            df["ticker"] = ticker.upper()

            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Stooq fetch failed for {ticker}: {e}")
            raise


def fetch_stooq(ticker: str, start: str = "2000-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to fetch from Stooq."""
    return StooqFetcher().fetch(ticker, start, end)
