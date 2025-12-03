"""
Yahoo Finance Fetcher - Stock, ETF, and market data

Uses yfinance library for fetching market data.
"""

import json
from pathlib import Path
from typing import Optional, Any, List, Dict
import pandas as pd
import logging

from fetch.fetcher_base import BaseFetcher


logger = logging.getLogger(__name__)

# Project root for finding registries
PROJECT_ROOT = Path(__file__).parent.parent


class YahooFetcher(BaseFetcher):
    """
    Fetcher for Yahoo Finance data.

    Supports stocks, ETFs, indices, currencies, and commodities.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize Yahoo fetcher.

        Args:
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(checkpoint_dir)

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
        Fetch data for a single Yahoo Finance ticker.

        Args:
            ticker: Yahoo ticker symbol (e.g., 'SPY', 'AAPL', '^VIX')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1wk', '1mo')

        Returns:
            DataFrame with OHLCV data
        """
        import yfinance as yf

        try:
            # Download data
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

            # Rename columns
            column_map = {
                "Date": "date",
                "Open": f"{ticker.lower()}_open",
                "High": f"{ticker.lower()}_high",
                "Low": f"{ticker.lower()}_low",
                "Close": ticker.lower(),  # Main column is just the ticker
                "Volume": f"{ticker.lower()}_volume"
            }
            df = df.rename(columns=column_map)

            return self.sanitize_dataframe(df, ticker)

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
            DataFrame with date and close price only
        """
        import yfinance as yf

        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            if not self.validate_response(df):
                return None

            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["date", ticker.lower()]

            return self.sanitize_dataframe(df, ticker)

        except Exception as e:
            logger.error(f"Yahoo error for {ticker}: {e}")
            return None

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
            DataFrame with date and all ticker close prices
        """
        import yfinance as yf

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
                # Get just Close prices
                df = df["Close"]

            df = df.reset_index()
            df.columns = ["date"] + [t.lower() for t in tickers]

            return df

        except Exception as e:
            logger.error(f"Yahoo batch error: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------------------
    # Registry and database integration
    # ----------------------------------------------------------------------
    def load_market_registry(self) -> List[Dict[str, Any]]:
        """Load the market registry."""
        registry_path = PROJECT_ROOT / "data" / "registry" / "market_registry.json"
        with open(registry_path, "r") as f:
            reg = json.load(f)
        return reg.get("instruments", [])

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch all enabled instruments from the market registry and write to database.

        Returns:
            Dictionary mapping instrument keys to DataFrames
        """
        from data.sql import prism_db

        instruments = self.load_market_registry()
        results = {}

        for instrument in instruments:
            if not instrument.get("enabled", True):
                logger.debug(f"Skipping disabled instrument: {instrument.get('key')}")
                continue

            key = instrument.get("key")
            ticker = instrument.get("ticker")
            frequency = instrument.get("frequency", "daily")

            logger.info(f"Fetching market instrument: {key} ({ticker})")

            try:
                df = self.fetch_single_close_only(ticker)

                if df is not None and not df.empty:
                    # Prepare data for database: need [date, value] columns
                    db_df = df.copy()
                    # Column is lowercase ticker, rename to 'value'
                    value_col = ticker.lower()
                    if value_col in db_df.columns:
                        db_df = db_df.rename(columns={value_col: "value"})
                    elif len(db_df.columns) == 2:  # date + one value column
                        value_col = [c for c in db_df.columns if c != "date"][0]
                        db_df = db_df.rename(columns={value_col: "value"})

                    # Register indicator and write to database
                    prism_db.add_indicator(key, system="market", frequency=frequency)
                    prism_db.write_dataframe(db_df, indicator=key, system="market")

                    results[key] = df
                    logger.info(f"  -> Wrote {len(df)} rows to database for {key}")
                else:
                    logger.warning(f"  -> No data returned for {key}")

            except Exception as e:
                logger.error(f"  -> Error fetching {key}: {e}")

        logger.info(f"Completed: {len(results)} market instruments fetched and stored")
        return results


# Common Yahoo Finance tickers for financial analysis
COMMON_YAHOO_TICKERS = {
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

    # Commodities
    "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures",
    "SI=F": "Silver Futures",

    # Currencies
    "DX-Y.NYB": "US Dollar Index",
    "EURUSD=X": "EUR/USD",
    "JPYUSD=X": "JPY/USD",

    # Bonds
    "TLT": "20+ Year Treasury ETF",
    "IEF": "7-10 Year Treasury ETF",
    "HYG": "High Yield Bond ETF",
}

# ---------------------------------------------------------------------
# Backward compatibility for older imports
# ---------------------------------------------------------------------
FetcherYahoo = YahooFetcher
