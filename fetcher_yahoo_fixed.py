"""
YahooFetcher - Fetch market data from Yahoo Finance
===================================================

This module provides a clean fetcher for Yahoo Finance market data.
It integrates with the unified database schema via data.sql.db.

Supports registries with Stooq as primary source by using fallback_ticker.

Usage:
    from fetch.fetcher_yahoo import YahooFetcher
    
    fetcher = YahooFetcher()
    
    # Fetch single ticker
    df = fetcher.fetch_single("SPY")
    
    # Fetch all enabled instruments from registry and write to database
    results = fetcher.fetch_all()
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from fetch.fetcher_base import BaseFetcher


logger = logging.getLogger(__name__)

# Project root for finding registries
PROJECT_ROOT = Path(__file__).parent.parent


class YahooFetcher(BaseFetcher):
    """
    Fetcher for Yahoo Finance data.

    Supports stocks, ETFs, indices, currencies, and commodities.
    
    Registry entry example:
    {
        "key": "spy",
        "ticker": "SPY.US",           # Stooq format
        "source": "stooq",
        "fallback_ticker": "SPY",     # Yahoo format
        "fallback_source": "yahoo",
        "enabled": true
    }
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize Yahoo fetcher.

        Args:
            checkpoint_dir: Optional directory for checkpoints
        """
        super().__init__(checkpoint_dir)
        self._yf = None

    def _get_yfinance(self):
        """Lazy import yfinance."""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    def validate_response(self, response: Any) -> bool:
        """Validate Yahoo Finance response."""
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
        interval: str = "1d",
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single Yahoo Finance ticker.

        Args:
            ticker: Yahoo ticker symbol (e.g., 'SPY', 'AAPL', '^VIX')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1wk', '1mo')

        Returns:
            DataFrame with date, value (close), and OHLCV columns
        """
        yf = self._get_yfinance()
        
        logger.info(f"Fetching Yahoo data: {ticker}")
        
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            
            if not self.validate_response(df):
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Reset index to get date as column
            df = df.reset_index()
            
            # Standardize column names
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "value",  # Main value is close price
                "Volume": "volume"
            })
            
            # Ensure date is datetime
            df["date"] = pd.to_datetime(df["date"])
            
            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Yahoo error for {ticker}: {e}")
            return None

    def fetch_single_close_only(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch only closing prices (lighter weight).

        Args:
            ticker: Yahoo ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date and value (close) columns only
        """
        df = self.fetch_single(ticker, start_date, end_date)
        
        if df is None:
            return None
        
        # Return only date and value columns
        return df[["date", "value"]].copy()

    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch multiple tickers efficiently in one call.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date index and ticker columns (close prices)
        """
        yf = self._get_yfinance()
        
        logger.info(f"Fetching {len(tickers)} tickers from Yahoo")
        
        try:
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            
            if not self.validate_response(df):
                return pd.DataFrame()
            
            # Handle MultiIndex columns from multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"]
            
            df = df.reset_index()
            df = df.rename(columns={"Date": "date"})
            
            # Lowercase column names
            df.columns = ["date"] + [t.lower() for t in tickers]
            
            return df
            
        except Exception as e:
            logger.error(f"Yahoo batch error: {e}")
            return pd.DataFrame()

    def load_market_registry(self) -> List[Dict[str, Any]]:
        """
        Load the market registry from data/registry/.
        
        Returns:
            List of instrument configurations
        """
        registry_path = PROJECT_ROOT / "data" / "registry" / "market_registry.json"
        
        if not registry_path.exists():
            logger.error(f"Market registry not found: {registry_path}")
            return []
        
        with open(registry_path, "r") as f:
            registry = json.load(f)
        
        return registry.get("instruments", [])

    def _get_yahoo_ticker(self, instrument: Dict[str, Any]) -> str:
        """
        Get the appropriate Yahoo ticker from an instrument config.
        
        Handles registries where Stooq is primary source by using fallback_ticker.
        
        Args:
            instrument: Instrument configuration dict
            
        Returns:
            Yahoo-compatible ticker string
        """
        source = instrument.get("source", "yahoo")
        
        # If source is yahoo, use ticker directly
        if source == "yahoo":
            return instrument.get("ticker", instrument.get("key", "").upper())
        
        # If source is stooq or other, try fallback_ticker first
        fallback_ticker = instrument.get("fallback_ticker")
        if fallback_ticker:
            return fallback_ticker
        
        # Last resort: try to convert Stooq ticker to Yahoo
        ticker = instrument.get("ticker", "")
        
        # Remove common Stooq suffixes
        if ticker.endswith(".US"):
            return ticker[:-3]  # SPY.US -> SPY
        if ticker.endswith(".F"):
            return ticker[:-2]  # GC.F -> GC
        
        return ticker

    def fetch_all(
        self,
        write_to_db: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all enabled instruments from the market registry.
        
        Args:
            write_to_db: If True, write results to database
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping instrument keys to DataFrames
        """
        instruments = self.load_market_registry()
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
        
        for instrument in instruments:
            if not instrument.get("enabled", True):
                logger.debug(f"Skipping disabled instrument: {instrument.get('key')}")
                continue
            
            key = instrument.get("key")
            
            # Get the correct Yahoo ticker (handles Stooq registries)
            ticker = self._get_yahoo_ticker(instrument)
            
            frequency = instrument.get("frequency", "daily")
            source = "yahoo"  # We're using Yahoo regardless of registry source
            
            logger.info(f"Fetching market instrument: {key} ({ticker})")
            
            try:
                df = self.fetch_single_close_only(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    results[key] = df
                    
                    if db:
                        # Register indicator
                        add_indicator(
                            name=key,
                            system="market",
                            frequency=frequency,
                            source=source,
                            description=instrument.get("name")
                        )
                        
                        # Write data
                        rows = write_dataframe(df, indicator=key, system="market")
                        
                        # Log fetch
                        log_fetch(
                            indicator=key,
                            system="market",
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
                            system="market",
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
                        system="market",
                        source=source,
                        rows_fetched=0,
                        status="error",
                        error_message=str(e)
                    )
        
        logger.info(f"Completed: {len(results)} market instruments fetched")
        return results

    def test_connection(self) -> bool:
        """
        Test the Yahoo Finance connection.
        
        Returns:
            True if connection successful
        """
        try:
            df = self.fetch_single("SPY")
            return df is not None and not df.empty
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Backward compatibility aliases
FetcherYahoo = YahooFetcher


# Common Yahoo Finance tickers reference
COMMON_TICKERS = {
    # Major Indices
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^VIX": "VIX Volatility",
    
    # Sector ETFs
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLU": "Utilities",
    
    # Bond ETFs
    "TLT": "20+ Year Treasury",
    "IEF": "7-10 Year Treasury",
    "SHY": "1-3 Year Treasury",
    "AGG": "Aggregate Bond",
    "HYG": "High Yield Bond",
    "LQD": "Investment Grade Corporate",
    
    # Commodities
    "GLD": "Gold ETF",
    "SLV": "Silver ETF",
    "USO": "Oil ETF",
    
    # Currency
    "DX-Y.NYB": "US Dollar Index",
    
    # Crypto
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}
