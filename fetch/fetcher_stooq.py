"""
Stooq Fetcher - Primary Market Data Source for PRISM

Reliable source for:
- US Equities and ETFs (SPY.US, QQQ.US)
- Commodity Futures (CL.F, GC.F, NG.F)
- Indices (SPX.US instead of ^GSPC)
"""

import pandas as pd
import requests
from io import StringIO
from typing import Optional, List
import logging
import time

from fetch.fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1  # seconds


class StooqFetcher(BaseFetcher):
    """
    Fetch daily OHLCV data from Stooq.

    Example ticker formatting:
        SPY.US      - US ETFs/Stocks
        QQQ.US
        CL.F        - Commodity Futures
        GC.F        - Gold Futures
        NG.F        - Natural Gas Futures

    Note: ^NDX does NOT work — must be NDX.US
    """

    BASE_URL = "https://stooq.com/q/d/l/"

    def validate_response(self, text: str) -> bool:
        """Validate CSV looks correct."""
        if not text or len(text.strip()) == 0:
            return False
        if "Date,Open,High,Low,Close" not in text:
            return False
        return True

    def fetch_single(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily data for a single Stooq ticker with retry logic.

        Args:
            ticker: Stooq ticker symbol (e.g., 'SPY.US', 'CL.F', 'GC.F')
            start_date: Optional start date (not used by Stooq API directly)
            end_date: Optional end date (not used by Stooq API directly)

        Returns:
            DataFrame with date and ticker column, or None if failed
        """
        url = f"{self.BASE_URL}?s={ticker.lower()}&i=d"

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, timeout=15)
                if response.status_code != 200:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY_BASE * (attempt + 1)
                        logger.debug(f"Stooq error {response.status_code} for {ticker}, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    logger.error(f"Stooq error {response.status_code} for {ticker}")
                    return None

                text = response.text
                if not self.validate_response(text):
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY_BASE * (attempt + 1)
                        logger.debug(f"Invalid CSV for {ticker}, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    logger.error(f"Invalid CSV for {ticker} after {MAX_RETRIES} attempts")
                    return None

                df = pd.read_csv(StringIO(text))
                if df.empty:
                    logger.warning(f"No rows returned for {ticker}")
                    return None

                # Stooq provides Date/Open/High/Low/Close/Volume
                df["date"] = pd.to_datetime(df["Date"])

                # Create ticker-named column for the close price
                ticker_col = ticker.lower().replace(".", "_")
                df[ticker_col] = df["Close"]

                # Filter by date range if specified
                if start_date:
                    df = df[df["date"] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df["date"] <= pd.to_datetime(end_date)]

                out = df[["date", ticker_col]].dropna()
                out = out.sort_values("date").reset_index(drop=True)

                logger.info(f"Stooq: {ticker} -> {len(out)} rows")
                return out

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (attempt + 1)
                    logger.debug(f"Stooq error for {ticker}: {e}, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Stooq error for {ticker} after {MAX_RETRIES} attempts: {e}")

        return None

    def fetch_all(
        self,
        registry: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch all Stooq tickers from the registry.

        IMPORTANT: This method ALWAYS returns a DataFrame, never None.

        Args:
            registry: PRISM metric registry dictionary
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with merged market data (empty if no data)
        """
        # Extract Stooq tickers from registry
        tickers = []
        for item in registry.get("market", []):
            source = item.get("source", "yahoo")
            if source != "stooq":
                continue

            # Get ticker from params or direct field
            ticker = (
                item.get("params", {}).get("ticker") or
                item.get("ticker") or
                item.get("symbol")
            )
            if ticker:
                tickers.append(ticker)

        if not tickers:
            logger.info("No Stooq tickers found in registry")
            return pd.DataFrame()

        logger.info(f"Fetching {len(tickers)} tickers from Stooq")

        # Fetch each ticker individually
        merged = None
        successful = 0
        for ticker in tickers:
            df = self.fetch_single(
                ticker,
                start_date=start_date,
                end_date=end_date
            )
            if df is None:
                logger.warning(f"Skipping Stooq ticker (no data): {ticker}")
                continue

            successful += 1
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="date", how="outer")

        logger.info(f"Stooq: Successfully fetched {successful}/{len(tickers)} tickers")

        return merged if merged is not None else pd.DataFrame()

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch multiple tickers and merge results.

        Args:
            tickers: List of Stooq ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and all ticker columns
        """
        if not tickers:
            return pd.DataFrame()

        merged = None
        for ticker in tickers:
            df = self.fetch_single(ticker, start_date=start_date, end_date=end_date)
            if df is None:
                continue

            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="date", how="outer")

        return merged if merged is not None else pd.DataFrame()
