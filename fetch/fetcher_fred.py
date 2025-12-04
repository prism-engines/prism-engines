"""
FREDFetcher - Fetch economic time series from FRED
==================================================

This module provides a clean fetcher for FRED (Federal Reserve Economic Data).
It integrates with the unified database schema via data.sql.db.

Usage:
    from fetch.fetcher_fred import FREDFetcher
    
    fetcher = FREDFetcher()
    
    # Fetch single series
    df = fetcher.fetch_single("DGS10")
    
    # Fetch all enabled series from registry and write to database
    results = fetcher.fetch_all()
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

    Registry entry example:
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

    def __init__(self, api_key: Optional[str] = None, checkpoint_dir: Optional[Path] = None):
        """
        Initialize FRED fetcher.
        
        Args:
            api_key: FRED API key (or set FRED_API_KEY env var)
            checkpoint_dir: Optional directory for checkpoints
        """
        super().__init__(checkpoint_dir)
        self.api_key = api_key or os.getenv("FRED_API_KEY") or os.getenv("FRED_API")
        
        if not self.api_key:
            logger.warning(
                "No FRED API key found. Set FRED_API_KEY environment variable. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def validate_response(self, response: Any) -> bool:
        """Validate FRED API response."""
        if response is None:
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
        Fetch data for a single FRED series.

        Args:
            ticker: The FRED series ID (e.g., 'DGS10')
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with 'date' and 'value' columns, or None on error
        """
        if not self.api_key:
            logger.error("Cannot fetch without FRED API key")
            return None
        
        params = {
            "series_id": ticker,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date or "1900-01-01",
        }
        
        if end_date:
            params["observation_end"] = end_date
        
        logger.info(f"Fetching FRED series: {ticker}")
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"FRED API error for {ticker}: {response.status_code}")
                return None
            
            data = response.json().get("observations", [])
            
            if not data:
                logger.warning(f"FRED returned no data for {ticker}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Clean up
            df = df.rename(columns={"date": "date", "value": "value"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Remove rows with invalid dates or values
            df = df.dropna(subset=["date", "value"])
            
            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            
            return df[["date", "value"]]
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple FRED series.
        
        Args:
            tickers: List of FRED series IDs
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        results = {}
        for ticker in tickers:
            df = self.fetch_single(ticker, start_date, end_date)
            if df is not None and not df.empty:
                results[ticker] = df
        return results

    def load_economic_registry(self) -> List[Dict[str, Any]]:
        """
        Load the economic registry from data/registry/.
        
        Returns:
            List of series configurations
        """
        registry_path = PROJECT_ROOT / "data" / "registry" / "economic_registry.json"
        
        if not registry_path.exists():
            logger.error(f"Economic registry not found: {registry_path}")
            return []
        
        with open(registry_path, "r") as f:
            registry = json.load(f)
        
        return registry.get("series", [])

    def fetch_all(
        self,
        write_to_db: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all enabled series from the economic registry.
        
        Args:
            write_to_db: If True, write results to database
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping indicator keys to DataFrames
        """
        series_list = self.load_economic_registry()
        results = {}
        
        # Import database module only if needed
        db = None
        if write_to_db:
            try:
                from data.sql.db import add_indicator, write_dataframe, log_fetch, init_database
                init_database()  # Ensure tables exist
                db = True
            except ImportError as e:
                logger.warning(f"Database module not available: {e}")
                db = None
        
        for series_config in series_list:
            if not series_config.get("enabled", True):
                logger.debug(f"Skipping disabled series: {series_config.get('key')}")
                continue
            
            key = series_config.get("key")
            ticker = series_config.get("ticker", key.upper())
            frequency = series_config.get("frequency", "daily")
            source = series_config.get("source", "fred")
            
            logger.info(f"Fetching economic series: {key} ({ticker})")
            
            try:
                df = self.fetch_single(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    results[key] = df
                    
                    if db:
                        # Register indicator
                        add_indicator(
                            name=key,
                            system="economic",
                            frequency=frequency,
                            source=source,
                            description=series_config.get("name")
                        )
                        
                        # Write data
                        rows = write_dataframe(df, indicator=key, system="economic")
                        
                        # Log fetch
                        log_fetch(
                            indicator=key,
                            system="economic",
                            source=source,
                            rows_fetched=rows,
                            status="success"
                        )
                        
                        logger.info(f"  -> Wrote {rows} rows to database for {key}")
                else:
                    logger.warning(f"  -> No data returned for {key}")
                    
                    if db:
                        log_fetch(
                            indicator=key,
                            system="economic",
                            source=source,
                            rows_fetched=0,
                            status="error",
                            error_message="No data returned"
                        )
                        
            except Exception as e:
                logger.error(f"  -> Error fetching {key}: {e}")
                
                if db:
                    log_fetch(
                        indicator=key,
                        system="economic",
                        source=source,
                        rows_fetched=0,
                        status="error",
                        error_message=str(e)
                    )
        
        logger.info(f"Completed: {len(results)} economic series fetched")
        return results

    def test_connection(self) -> bool:
        """
        Test the FRED API connection.
        
        Returns:
            True if connection successful
        """
        if not self.api_key:
            logger.error("No API key configured")
            return False
        
        try:
            df = self.fetch_single("DGS10")
            return df is not None and not df.empty
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Backward compatibility aliases
FetcherFRED = FREDFetcher
