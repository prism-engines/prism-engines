"""
Stooq Fetcher - Primary Market Data Source for PRISM
"""

import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from typing import Optional
import logging

from fetch.fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)


class StooqFetcher(BaseFetcher):
    """
    Fetch daily OHLCV data from Stooq.

    Example ticker formatting:
        SPY.US
        QQQ.US
        IWM.US
        ^NDX does NOT work — must be NDX.US
    """

    BASE_URL = "https://stooq.com/q/d/l/"

    def validate_response(self, text: str) -> bool:
        """Validate CSV looks correct."""
        if not text or len(text.strip()) == 0:
            return False
        if "Date,Open,High,Low,Close,Volume" not in text:
            return False
        return True

    def fetch_single(self, ticker: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch daily data for a single Stooq ticker.

        Returns DataFrame with:
            date, value
        where value = Close
        """
        url = f"{self.BASE_URL}?s={ticker}&i=d"
        logger.info(f"Stooq request → {url}")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Stooq error {response.status_code} for {ticker}")
                return None

            text = response.text
            if not self.validate_response(text):
                logger.error(f"Invalid CSV for {ticker}")
                return None

            df = pd.read_csv(StringIO(text))
            if df.empty:
                logger.warning(f"No rows returned for {ticker}")
                return None

            # Stooq provides Date/Open/High/Low/Close/Volume
            df["date"] = pd.to_datetime(df["Date"])
            df["value"] = df["Close"]

            out = df[["date", "value"]].dropna()

            logger.info(f"Stooq returned {len(out)} rows for {ticker}")
            return out

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None