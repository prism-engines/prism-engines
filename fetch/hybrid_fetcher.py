"""
Hybrid Fetcher - Stooq primary with Tiingo fallback
====================================================

Strategy:
1. Try Stooq first (free, CSV-based, reliable for US securities)
2. Fall back to Tiingo if Stooq fails (official API, reliable)
3. For crypto: Use Tiingo directly (Stooq doesn't have crypto)

This replaces the old Yahoo fallback which was unreliable due to scraping.
"""

import json
import os
from pathlib import Path
import logging
from typing import Optional

import pandas as pd

from fetch.fetcher_stooq import StooqFetcher
from fetch.fetcher_tiingo import TiingoFetcher

logger = logging.getLogger(__name__)

# Path to fallback JSON map (maps Stooq tickers to Tiingo tickers if different)
FALLBACK_MAP_PATH = Path("data/registry/tiingo_fallback_map.json")

# Default Tiingo API key (can be overridden via environment or constructor)
DEFAULT_TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY", "39a87d4f616a7432dbc92533eeed14c238f6d159")


def load_fallback_map() -> dict:
    """Load the Tiingo fallback ticker map."""
    if not FALLBACK_MAP_PATH.exists():
        logger.debug(f"No Tiingo fallback map found: {FALLBACK_MAP_PATH}")
        return {}

    with open(FALLBACK_MAP_PATH) as f:
        return json.load(f)


class HybridFetcher:
    """
    Hybrid Stooq -> Tiingo fallback fetcher.

    Primary: Stooq (free, reliable for US ETFs/stocks)
    Fallback: Tiingo (official API, reliable)
    Crypto: Tiingo directly

    Usage:
        fetcher = HybridFetcher()
        df = fetcher.fetch_single("SPY.US")  # Tries Stooq, falls back to Tiingo
        df = fetcher.fetch_crypto("btcusd")   # Uses Tiingo crypto endpoint
    """

    def __init__(self, tiingo_api_key: Optional[str] = None):
        """
        Initialize hybrid fetcher.

        Args:
            tiingo_api_key: Tiingo API key. Falls back to env var or default.
        """
        self.stooq = StooqFetcher()

        api_key = tiingo_api_key or DEFAULT_TIINGO_API_KEY
        self.tiingo = TiingoFetcher(api_key=api_key)

        self.fallback_map = load_fallback_map()

    def resolve_tiingo_ticker(self, ticker: str) -> str:
        """
        Resolve Stooq ticker to Tiingo ticker.

        First checks fallback map, then strips Stooq suffixes.
        """
        # Check explicit mapping first
        mapped = self.fallback_map.get(ticker.upper())
        if mapped:
            return mapped

        # Strip Stooq suffixes
        clean = ticker.upper()
        if clean.endswith(".US"):
            return clean[:-3]
        if clean.endswith(".F"):
            return clean[:-2]

        return clean

    def fetch_single(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch market data with Stooq primary, Tiingo fallback.

        Args:
            ticker: Ticker symbol (Stooq format like SPY.US, or plain like SPY)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date and price column, or None if both fail
        """
        # Try Stooq first
        df_stooq = self.stooq.fetch_single(
            ticker,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

        if df_stooq is not None and len(df_stooq) > 0:
            logger.info(f"[HybridFetcher] Stooq OK -> {ticker}")
            return df_stooq

        logger.warning(f"[HybridFetcher] Stooq FAIL -> {ticker}")

        # Fallback to Tiingo
        tiingo_ticker = self.resolve_tiingo_ticker(ticker)
        logger.info(f"[HybridFetcher] Falling back to Tiingo -> {tiingo_ticker}")

        df_tiingo = self.tiingo.fetch_single(
            tiingo_ticker,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

        if df_tiingo is None or df_tiingo.empty:
            logger.error(f"[HybridFetcher] Tiingo FAIL -> {tiingo_ticker}")
            return None

        logger.info(f"[HybridFetcher] Tiingo OK -> {tiingo_ticker}")
        return df_tiingo

    def fetch_crypto(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch cryptocurrency data via Tiingo.

        Args:
            ticker: Crypto ticker (e.g., 'btcusd', 'ethusd', 'BTC-USD')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and price column
        """
        return self.tiingo.fetch_crypto(
            ticker,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

    def fetch_all_from_registry(
        self,
        registry: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch all instruments from a registry.

        Handles both market data (Stooq/Tiingo) and crypto (Tiingo).

        Args:
            registry: PRISM registry dictionary with 'instruments' list
            start_date: Start date
            end_date: End date

        Returns:
            Merged DataFrame with all instruments
        """
        instruments = registry.get("instruments", [])
        if not instruments:
            logger.warning("No instruments in registry")
            return pd.DataFrame()

        merged = None
        success_count = 0

        for inst in instruments:
            if not inst.get("enabled", True):
                continue

            key = inst.get("key", "")
            ticker = inst.get("ticker", "")
            asset_class = inst.get("asset_class", "")

            logger.info(f"Fetching {key} ({ticker})...")

            try:
                # Use crypto endpoint for crypto assets
                if asset_class == "crypto":
                    df = self.fetch_crypto(
                        ticker,
                        start_date=start_date,
                        end_date=end_date
                    )
                else:
                    df = self.fetch_single(
                        ticker,
                        start_date=start_date,
                        end_date=end_date
                    )

                if df is not None and not df.empty:
                    # Rename column to use registry key
                    cols = [c for c in df.columns if c != "date"]
                    if cols:
                        df = df.rename(columns={cols[0]: key})

                    success_count += 1

                    if merged is None:
                        merged = df
                    else:
                        merged = pd.merge(merged, df, on="date", how="outer")
                else:
                    logger.warning(f"  -> No data for {key}")

            except Exception as e:
                logger.error(f"  -> Error fetching {key}: {e}")

        logger.info(f"Fetched {success_count}/{len([i for i in instruments if i.get('enabled', True)])} instruments")

        return merged if merged is not None else pd.DataFrame()

    def test_connection(self) -> dict:
        """
        Test connections to both Stooq and Tiingo.

        Returns:
            Dictionary with status of each source
        """
        results = {
            "stooq": False,
            "tiingo": False,
            "tiingo_crypto": False
        }

        # Test Stooq
        try:
            df = self.stooq.fetch_single("SPY.US")
            results["stooq"] = df is not None and not df.empty
        except Exception as e:
            logger.error(f"Stooq test failed: {e}")

        # Test Tiingo EOD
        try:
            df = self.tiingo.fetch_single("SPY")
            results["tiingo"] = df is not None and not df.empty
        except Exception as e:
            logger.error(f"Tiingo test failed: {e}")

        # Test Tiingo Crypto
        try:
            df = self.tiingo.fetch_crypto("btcusd")
            results["tiingo_crypto"] = df is not None and not df.empty
        except Exception as e:
            logger.error(f"Tiingo crypto test failed: {e}")

        return results
