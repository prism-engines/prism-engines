"""
Tiingo Fetcher - Reliable Market Data Source for PRISM
=======================================================

Tiingo provides:
- US Stocks and ETFs (daily EOD data)
- Crypto prices (BTC, ETH, etc.)
- Official REST API (no scraping)
- Free tier: 500 requests/day

Documentation: https://api.tiingo.com/documentation
"""

import pandas as pd
import requests
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import time

from fetch.fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1  # seconds

# Tiingo API endpoints
TIINGO_EOD_URL = "https://api.tiingo.com/tiingo/daily"
TIINGO_CRYPTO_URL = "https://api.tiingo.com/tiingo/crypto/prices"


class TiingoFetcher(BaseFetcher):
    """
    Fetcher for Tiingo financial data.

    Supports:
    - US stocks and ETFs via EOD endpoint
    - Cryptocurrencies via crypto endpoint

    Example usage:
        fetcher = TiingoFetcher(api_key="your_key")
        df = fetcher.fetch_single("SPY")
        df_crypto = fetcher.fetch_crypto("btcusd")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize Tiingo fetcher.

        Args:
            api_key: Tiingo API key (required)
            checkpoint_dir: Optional checkpoint directory
        """
        super().__init__(checkpoint_dir)
        self.api_key = api_key

        if not self.api_key:
            raise ValueError("Tiingo API key is required")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}"
        }

    def validate_response(self, response: Any) -> bool:
        """Validate Tiingo API response."""
        if response is None:
            return False
        if isinstance(response, list) and len(response) == 0:
            return False
        if isinstance(response, pd.DataFrame) and response.empty:
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
        Fetch daily EOD data for a single ticker.

        Args:
            ticker: Stock/ETF ticker symbol (e.g., 'SPY', 'QQQ')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date and price columns, or None on failure
        """
        # Clean ticker - remove common suffixes from other sources
        clean_ticker = self._clean_ticker(ticker)

        url = f"{TIINGO_EOD_URL}/{clean_ticker}/prices"

        params = {}
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=15
                )

                if response.status_code == 404:
                    logger.warning(f"Tiingo: Ticker not found: {clean_ticker}")
                    return None

                if response.status_code == 401:
                    logger.error("Tiingo: Invalid API key")
                    return None

                if response.status_code == 429:
                    # Rate limited
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Tiingo rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    continue

                if response.status_code != 200:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY_BASE * (attempt + 1)
                        logger.debug(f"Tiingo error {response.status_code}, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    logger.error(f"Tiingo error {response.status_code} for {clean_ticker}")
                    return None

                data = response.json()

                if not self.validate_response(data):
                    logger.warning(f"Tiingo: No data for {clean_ticker}")
                    return None

                df = pd.DataFrame(data)

                # Standardize columns
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

                # Create ticker-named column for close price
                ticker_col = clean_ticker.lower().replace("-", "_")
                df[ticker_col] = df["adjClose"]  # Use adjusted close

                # Filter columns
                out = df[["date", ticker_col]].copy()
                out = out.sort_values("date").reset_index(drop=True)

                logger.info(f"Tiingo: {clean_ticker} -> {len(out)} rows")
                return out

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (attempt + 1)
                    logger.debug(f"Tiingo request error: {e}, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Tiingo request failed for {clean_ticker}: {e}")
            except Exception as e:
                last_error = e
                logger.error(f"Tiingo error for {clean_ticker}: {e}")
                break

        return None

    def fetch_crypto(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch cryptocurrency price data.

        Args:
            ticker: Crypto ticker (e.g., 'btcusd', 'ethusd', 'BTC-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date and price columns, or None on failure
        """
        # Convert Yahoo-style crypto tickers to Tiingo format
        clean_ticker = self._clean_crypto_ticker(ticker)

        params = {
            "tickers": clean_ticker,
            "resampleFreq": "1day"
        }
        if start_date:
            params["startDate"] = start_date
        if end_date:
            params["endDate"] = end_date

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
                    TIINGO_CRYPTO_URL,
                    headers=self.headers,
                    params=params,
                    timeout=15
                )

                if response.status_code == 429:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Tiingo rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    continue

                if response.status_code != 200:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY_BASE * (attempt + 1)
                        time.sleep(delay)
                        continue
                    logger.error(f"Tiingo crypto error {response.status_code} for {clean_ticker}")
                    return None

                data = response.json()

                if not data or len(data) == 0:
                    logger.warning(f"Tiingo: No crypto data for {clean_ticker}")
                    return None

                # Tiingo returns nested structure for crypto
                price_data = data[0].get("priceData", [])
                if not price_data:
                    logger.warning(f"Tiingo: Empty crypto price data for {clean_ticker}")
                    return None

                df = pd.DataFrame(price_data)

                # Standardize
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

                # Use 'close' for crypto
                ticker_col = clean_ticker.lower()
                df[ticker_col] = df["close"]

                out = df[["date", ticker_col]].copy()
                out = out.sort_values("date").reset_index(drop=True)

                logger.info(f"Tiingo crypto: {clean_ticker} -> {len(out)} rows")
                return out

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE * (attempt + 1)
                    time.sleep(delay)
                else:
                    logger.error(f"Tiingo crypto request failed for {clean_ticker}: {e}")
            except Exception as e:
                logger.error(f"Tiingo crypto error for {clean_ticker}: {e}")
                break

        return None

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
            tickers: List of ticker symbols
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

    def _clean_ticker(self, ticker: str) -> str:
        """
        Clean ticker symbol for Tiingo API.

        Removes Stooq suffixes like .US, .F
        """
        ticker = ticker.upper()

        # Remove Stooq suffixes
        if ticker.endswith(".US"):
            return ticker[:-3]
        if ticker.endswith(".F"):
            return ticker[:-2]

        return ticker

    def _clean_crypto_ticker(self, ticker: str) -> str:
        """
        Convert crypto ticker to Tiingo format.

        Yahoo: BTC-USD -> Tiingo: btcusd
        """
        ticker = ticker.lower()

        # Remove dash for Yahoo-style tickers
        ticker = ticker.replace("-", "")

        return ticker

    def test_connection(self) -> bool:
        """
        Test the Tiingo API connection.

        Returns:
            True if connection successful
        """
        try:
            df = self.fetch_single("SPY")
            return df is not None and not df.empty
        except Exception as e:
            logger.error(f"Tiingo connection test failed: {e}")
            return False


# Backward compatibility alias
FetcherTiingo = TiingoFetcher
