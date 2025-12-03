"""
FREDFetcher - Fetch economic time series from FRED

This file replaces all previous versions and removes:
✓ circular imports
✓ self-imports
✓ broken class names
✓ duplicate loaders

It integrates cleanly with BaseFetcher and the new registry format.
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import requests

from fetch.fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)

# Project root for finding registries
PROJECT_ROOT = Path(__file__).parent.parent


class FREDFetcher(BaseFetcher):
    """
    Fetch economic data from FRED (Federal Reserve Bank of St. Louis).

    registry entry example:
    {
        "key": "dgs10",
        "ticker": "DGS10",
        "source": "fred",
        "enabled": true,
        "name": "10-Year Treasury Constant Maturity Rate",
        "frequency": "daily"
    }
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("FRED_API_KEY")

        if not self.api_key:
            logger.warning("No FRED API key found. FRED requests may fail.")

    # ----------------------------------------------------------------------
    # Core fetch function
    # ----------------------------------------------------------------------
    def fetch(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch a FRED series defined in the registry.
        """

        series_id = config.get("ticker")
        if not series_id:
            raise ValueError("Economic registry entry missing 'ticker'")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": "1900-01-01",
        }

        logger.info(f"Fetching FRED series: {series_id}")

        response = requests.get(self.BASE_URL, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch {series_id} from FRED: "
                f"{response.status_code} {response.text}"
            )

        data = response.json().get("observations", [])
        if not data:
            raise RuntimeError(f"FRED returned no data for {series_id}")

        df = pd.DataFrame(data)
        df = df.rename(columns={"date": "timestamp", "value": series_id})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

        df = df.dropna(subset=["timestamp"])

        df = df.set_index("timestamp").sort_index()

        logger.info(f"Fetched {len(df)} rows for {series_id}")

        return df

    # ----------------------------------------------------------------------
    # Abstract method implementations
    # ----------------------------------------------------------------------
    def fetch_single(self, ticker: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single FRED series.

        Args:
            ticker: The FRED series ID (e.g., 'DGS10')

        Returns:
            DataFrame with 'date' column and ticker data
        """
        config = {"ticker": ticker}
        try:
            df = self.fetch(config)
            # Reset index to get date as column
            df = df.reset_index()
            df = df.rename(columns={"timestamp": "date"})
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def validate_response(self, response: Any) -> bool:
        """Validate FRED API response."""
        if response is None:
            return False
        if isinstance(response, pd.DataFrame) and response.empty:
            return False
        return True

    # ----------------------------------------------------------------------
    # Registry and database integration
    # ----------------------------------------------------------------------
    def load_economic_registry(self) -> List[Dict[str, Any]]:
        """Load the economic registry."""
        registry_path = PROJECT_ROOT / "data" / "registry" / "economic_registry.json"
        with open(registry_path, "r") as f:
            reg = json.load(f)
        return reg.get("series", [])

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all enabled series from the economic registry and write to database.

        Returns:
            Dictionary mapping indicator keys to DataFrames
        """
        from data.sql import prism_db

        series_list = self.load_economic_registry()
        results = {}

        for series_config in series_list:
            if not series_config.get("enabled", True):
                logger.debug(f"Skipping disabled series: {series_config.get('key')}")
                continue

            key = series_config.get("key")
            ticker = series_config.get("ticker")
            frequency = series_config.get("frequency", "daily")

            logger.info(f"Fetching economic series: {key} ({ticker})")

            try:
                df = self.fetch(series_config)

                if df is not None and not df.empty:
                    # Reset index to get date as column
                    df = df.reset_index()
                    df = df.rename(columns={"timestamp": "date", ticker: "value"})

                    # Register indicator and write to database
                    prism_db.add_indicator(key, system="economic", frequency=frequency)
                    prism_db.write_dataframe(df, indicator=key, system="economic")

                    results[key] = df
                    logger.info(f"  -> Wrote {len(df)} rows to database for {key}")
                else:
                    logger.warning(f"  -> No data returned for {key}")

            except Exception as e:
                logger.error(f"  -> Error fetching {key}: {e}")

        logger.info(f"Completed: {len(results)} economic series fetched and stored")
        return results
