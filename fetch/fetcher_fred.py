"""
FRED Fetcher - Federal Reserve Economic Data
"""

import os
from pathlib import Path
from typing import Optional, Any
import pandas as pd
import logging

from .fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)


class FREDFetcher(BaseFetcher):
    """
    Fetcher for Federal Reserve Economic Data (FRED).

    Requires FRED_API environment variable or Colab secret.
    """

    def __init__(self, api_key: Optional[str] = None, checkpoint_dir: Optional[Path] = None):
        """
        Initialize FRED fetcher.

        Args:
            api_key: FRED API key (or set FRED_API env var)
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(checkpoint_dir)
        self.api_key = api_key
        self.fred = None

    def _init_client(self) -> None:
        """Initialize FRED API client."""
        if self.fred is not None:
            return

        from fredapi import Fred

        api_key = self.api_key

        # Try Colab secrets first
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get("FRED_API")
            except (ImportError, Exception):
                pass

        # Fallback to environment variable
        if not api_key:
            api_key = os.environ.get("FRED_API")

        if not api_key:
            raise ValueError(
                "FRED API key not found. Set FRED_API environment variable "
                "or pass api_key parameter."
            )

        self.fred = Fred(api_key=api_key)
        logger.info("FRED client initialized")

    def validate_response(self, response: Any) -> bool:
        """Validate FRED API response."""
        if response is None:
            return False
        if isinstance(response, pd.Series) and response.empty:
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
        Fetch a single FRED series.

        Args:
            ticker: FRED series ID (e.g., 'GDP', 'UNRATE', 'DFF')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date and value columns
        """
        self._init_client()

        try:
            # Fetch series
            series = self.fred.get_series(
                ticker,
                observation_start=start_date,
                observation_end=end_date
            )

            if not self.validate_response(series):
                logger.warning(f"Invalid response for {ticker}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame({
                "date": series.index,
                ticker.lower(): series.values
            })

            return self.sanitize_dataframe(df, ticker)

        except Exception as e:
            logger.error(f"FRED error for {ticker}: {e}")
            return None

    def get_series_info(self, ticker: str) -> Optional[dict]:
        """
        Get metadata for a FRED series.

        Args:
            ticker: FRED series ID

        Returns:
            Dictionary with series metadata
        """
        self._init_client()

        try:
            info = self.fred.get_series_info(ticker)
            return info.to_dict() if info is not None else None
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
            return None


# Common FRED series for financial analysis
COMMON_FRED_SERIES = {
    # Interest Rates
    "DFF": "Federal Funds Rate",
    "DGS10": "10-Year Treasury",
    "DGS2": "2-Year Treasury",
    "T10Y2Y": "10Y-2Y Spread",

    # Economic Indicators
    "GDP": "Gross Domestic Product",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "PCEPI": "PCE Price Index",

    # Money Supply
    "M2SL": "M2 Money Supply",
    "BOGMBASE": "Monetary Base",

    # Credit
    "BAMLH0A0HYM2": "High Yield Spread",
    "TEDRATE": "TED Spread",

    # Housing
    "CSUSHPINSA": "Case-Shiller Home Price Index",
    "HOUST": "Housing Starts",

    # Manufacturing
    "INDPRO": "Industrial Production",
    "UMCSENT": "Consumer Sentiment",
}

# ---------------------------------------------------------------------
# Backward compatibility for older imports
# ---------------------------------------------------------------------
FetcherFRED = FREDFetcher
