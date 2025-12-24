"""
PRISM Tiingo Fetcher

Fetches market data from the Tiingo API.

This fetcher:
    - Returns raw data as DataFrames
    - Does NOT write to database
    - Does NOT transform values
    
API Documentation: https://www.tiingo.com/documentation/
"""

import os
from datetime import date
from typing import Optional
import pandas as pd
import requests
import logging

from .base import BaseFetcher, FetchResult


logger = logging.getLogger(__name__)


class TiingoFetcher(BaseFetcher):
    """
    Fetcher for Tiingo market data.
    
    Supports:
        - Daily stock prices (end-of-day)
        - ETF prices
        
    Requires TIINGO_API_KEY environment variable or pass api_key to constructor.
    
    Usage:
        fetcher = TiingoFetcher()
        result = fetcher.fetch("AAPL")
        if result.success:
            df = result.data  # DataFrame with [date, value]
    """
    
    BASE_URL = "https://api.tiingo.com/tiingo/daily"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tiingo fetcher.
        
        Args:
            api_key: Tiingo API key. If not provided, reads from TIINGO_API_KEY env var.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Tiingo API key required. Set TIINGO_API_KEY environment variable "
                "or pass api_key to constructor."
            )
    
    @property
    def source_name(self) -> str:
        return "tiingo"
    
    def fetch(
        self,
        indicator_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        price_field: str = "adjClose",
    ) -> FetchResult:
        """
        Fetch daily price data from Tiingo.
        
        Args:
            indicator_id: Ticker symbol (e.g., 'AAPL', 'SPY', 'XLU')
            start_date: Optional start date (defaults to 1990-01-01)
            end_date: Optional end date
            price_field: Which price to use as 'value'. Default 'adjClose'.
                        Options: 'open', 'high', 'low', 'close', 'adjClose'
            
        Returns:
            FetchResult with DataFrame containing [date, value]
        """
        try:
            # Build URL
            url = f"{self.BASE_URL}/{indicator_id}/prices"
            
            # Headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Token {self.api_key}",
            }
            
            # Parameters
            params = {
                "startDate": start_date.isoformat() if start_date else "1990-01-01",
                "resampleFreq": "daily",
            }
            if end_date:
                params["endDate"] = end_date.isoformat()
            
            # Make request
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            if not data:
                return FetchResult.from_error(indicator_id, "No data returned")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Tiingo returns 'date' and various price fields
            if "date" not in df.columns:
                return FetchResult.from_error(indicator_id, "No 'date' column in response")
                
            if price_field not in df.columns:
                available = [c for c in df.columns if c != "date"]
                return FetchResult.from_error(
                    indicator_id, 
                    f"Price field '{price_field}' not found. Available: {available}"
                )
            
            # Select date and chosen price field, rename to standard columns
            df = df[["date", price_field]].copy()
            df = df.rename(columns={price_field: "value"})
            
            # Convert value to numeric
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Drop NaN values
            df = df.dropna(subset=["value"])
            
            if df.empty:
                return FetchResult.from_error(indicator_id, "All values were missing")
            
            # Validate and standardize
            df = self.validate_dataframe(df, indicator_id)
            
            return FetchResult.from_dataframe(indicator_id, df)
            
        except requests.exceptions.Timeout:
            return FetchResult.from_error(indicator_id, "Request timed out")
        except requests.exceptions.HTTPError as e:
            # Tiingo returns 404 for unknown tickers
            if e.response.status_code == 404:
                return FetchResult.from_error(indicator_id, f"Ticker not found: {indicator_id}")
            return FetchResult.from_error(indicator_id, f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            return FetchResult.from_error(indicator_id, f"Request error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error fetching {indicator_id}")
            return FetchResult.from_error(indicator_id, f"Unexpected error: {e}")