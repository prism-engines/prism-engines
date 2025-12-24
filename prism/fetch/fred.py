"""
PRISM FRED Fetcher

Fetches economic data from the Federal Reserve Economic Data (FRED) API.

This fetcher:
    - Returns raw data as DataFrames
    - Does NOT write to database
    - Does NOT transform values
    
API Documentation: https://fred.stlouisfed.org/docs/api/
"""

import os
from datetime import date
from typing import Optional
import pandas as pd
import requests
import logging

from .base import BaseFetcher, FetchResult


logger = logging.getLogger(__name__)


class FredFetcher(BaseFetcher):
    """
    Fetcher for Federal Reserve Economic Data (FRED).
    
    Requires FRED_API_KEY environment variable or pass api_key to constructor.
    
    Usage:
        fetcher = FredFetcher()
        result = fetcher.fetch("GDP")
        if result.success:
            df = result.data  # DataFrame with [date, value]
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED fetcher.
        
        Args:
            api_key: FRED API key. If not provided, reads from FRED_API_KEY env var.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key to constructor."
            )
    
    @property
    def source_name(self) -> str:
        return "fred"
    
    def fetch(
        self,
        indicator_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> FetchResult:
        """
        Fetch a single series from FRED.
        
        Args:
            indicator_id: FRED series ID (e.g., 'GDP', 'UNRATE', 'DGS10')
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            FetchResult with DataFrame containing [date, value]
        """
        try:
            # Build request parameters
            params = {
                "series_id": indicator_id,
                "api_key": self.api_key,
                "file_type": "json",
            }
            
            if start_date:
                params["observation_start"] = start_date.isoformat()
            if end_date:
                params["observation_end"] = end_date.isoformat()
            
            # Make request
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            if "observations" not in data:
                return FetchResult.from_error(
                    indicator_id, 
                    f"No observations in response: {data.get('error_message', 'Unknown error')}"
                )
            
            observations = data["observations"]
            
            if not observations:
                return FetchResult.from_error(indicator_id, "No data returned")
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            
            # FRED returns 'date' and 'value' columns
            # Values may be '.' for missing data
            df = df[["date", "value"]].copy()
            
            # Convert value to numeric, coercing errors to NaN
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Drop rows where value is NaN (FRED uses '.' for missing)
            df = df.dropna(subset=["value"])
            
            if df.empty:
                return FetchResult.from_error(indicator_id, "All values were missing")
            
            # Validate and standardize
            df = self.validate_dataframe(df, indicator_id)
            
            return FetchResult.from_dataframe(indicator_id, df)
            
        except requests.exceptions.Timeout:
            return FetchResult.from_error(indicator_id, "Request timed out")
        except requests.exceptions.HTTPError as e:
            return FetchResult.from_error(indicator_id, f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            return FetchResult.from_error(indicator_id, f"Request error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error fetching {indicator_id}")
            return FetchResult.from_error(indicator_id, f"Unexpected error: {e}")
