"""
YahooFetcher - Fetch market data from Yahoo Finance
===================================================

This module provides a clean fetcher for Yahoo Finance market data.
It integrates with the unified database schema via data.sql.db.

FIXED: Handles MultiIndex columns from yfinance >= 0.2.40
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from fetch.fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent


class YahooFetcher(BaseFetcher):
    """
    Fetcher for Yahoo Finance data.

    Supports stocks, ETFs, indices, currencies, and commodities.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        super().__init__(checkpoint_dir)
        self._yf = None

    def _get_yfinance(self):
        """Lazy import so import errors don't break everything."""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    # ----------------------------------------------------------------------
    # COLUMN FLATTENING (handles MultiIndex from newer yfinance)
    # ----------------------------------------------------------------------
    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten MultiIndex columns that yfinance returns.
        
        yfinance >= 0.2.40 returns columns like:
            ('Close', 'SPY'), ('Open', 'SPY'), etc.
        
        We need to flatten to just:
            'Close', 'Open', etc.
        """
        if isinstance(df.columns, pd.MultiIndex):
            # Take just the first level (the metric name)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df

    # ----------------------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------------------
    def validate_response(self, response: Any) -> bool:
        """Return True only if response is a non-empty DataFrame."""
        if response is None:
            return False
        if isinstance(response, pd.DataFrame) and response.empty:
            return False
        return True

    # ----------------------------------------------------------------------
    # SINGLE-TICKER FETCH
    # ----------------------------------------------------------------------
    def fetch_single(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs
    ) -> Optional[pd.DataFrame]:

        yf = self._get_yfinance()
        logger.info(f"Fetching Yahoo data: {ticker}")

        try:
        import pandas_datareader.data as pdr
        df = pdr.DataReader(ticker, 'stooq')
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False
            )

            # Must use explicit check
            if df is None or df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            # CRITICAL FIX: Flatten MultiIndex columns BEFORE reset_index
            df = self._flatten_columns(df)
            
            # Now reset index
            df = df.reset_index()
            
            # Flatten again in case Date became a tuple
            df = self._flatten_columns(df)

            # Now rename works correctly
            rename_map = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if col_lower == "date":
                    rename_map[col] = "date"
                elif col_lower == "close":
                    rename_map[col] = "value"
                elif col_lower == "open":
                    rename_map[col] = "open"
                elif col_lower == "high":
                    rename_map[col] = "high"
                elif col_lower == "low":
                    rename_map[col] = "low"
                elif col_lower == "volume":
                    rename_map[col] = "volume"
                elif col_lower == "adj close":
                    rename_map[col] = "adj_close"
            
            df = df.rename(columns=rename_map)

            # Ensure date is datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Yahoo error for {ticker}: {e}")
            return None

    # ----------------------------------------------------------------------
    # SINGLE-TICKER CLOSE-ONLY
    # ----------------------------------------------------------------------
    def fetch_single_close_only(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:

        df = self.fetch_single(ticker, start_date, end_date)
        if df is None or df.empty:
            return None

        # Make sure we have the expected columns
        if "date" not in df.columns or "value" not in df.columns:
            logger.error(f"Missing expected columns. Got: {df.columns.tolist()}")
            return None

        return df[["date", "value"]].copy()

    # ----------------------------------------------------------------------
    # MULTI-TICKER FETCH
    # ----------------------------------------------------------------------
    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:

        yf = self._get_yfinance()
        logger.info(f"Fetching {len(tickers)} tickers from Yahoo")

        try:
        import pandas_datareader.data as pdr
        df = pdr.DataReader(ticker, 'stooq')
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # Handle MultiIndex columns - get just Close prices
            if isinstance(df.columns, pd.MultiIndex):
                # For multiple tickers, columns are ('Close', 'SPY'), ('Close', 'AAPL'), etc.
                # We want just Close prices, pivoted by ticker
                if 'Close' in df.columns.get_level_values(0):
                    df = df['Close']
                elif 'Adj Close' in df.columns.get_level_values(0):
                    df = df['Adj Close']
            
            df = df.reset_index()
            
            # Flatten any remaining MultiIndex
            df = self._flatten_columns(df)
            
            # Rename Date column
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            
            # Lowercase ticker columns
            df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

            return df

        except Exception as e:
            logger.error(f"Yahoo batch error: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------------------
    # REGISTRY
    # ----------------------------------------------------------------------
    def load_market_registry(self) -> List[Dict[str, Any]]:
        registry_path = PROJECT_ROOT / "data" / "registry" / "market_registry.json"

        if not registry_path.exists():
            logger.error(f"Market registry not found: {registry_path}")
            return []

        with open(registry_path, "r") as f:
            registry = json.load(f)

        return registry.get("instruments", [])

    # ----------------------------------------------------------------------
    # FETCH ALL
    # ----------------------------------------------------------------------
    def fetch_all(
        self,
        write_to_db: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:

        instruments = self.load_market_registry()
        results = {}

        db_enabled = False
        if write_to_db:
            try:
                from data.sql.db import add_indicator, write_dataframe, log_fetch, init_database
                init_database()
                db_enabled = True
            except Exception as e:
                logger.error(f"DB unavailable: {e}")

        for instrument in instruments:
            if not instrument.get("enabled", True):
                continue

            key = instrument["key"]
            ticker = instrument.get("ticker_yahoo", instrument.get("ticker", key.upper()))
            frequency = instrument.get("frequency", "daily")
            source = instrument.get("source", "yahoo")

            logger.info(f"Fetching market instrument: {key} ({ticker})")

            try:
                df = self.fetch_single_close_only(ticker, start_date, end_date)

                if df is None or df.empty:
                    logger.warning(f"  -> No data returned for {key}")

                    if db_enabled:
                        log_fetch(
                            indicator=key,
                            system="market",
                            source=source,
                            rows_fetched=0,
                            status="error",
                            error_message="No data returned"
                        )
                    continue

                # Valid data
                results[key] = df

                if db_enabled:
                    add_indicator(
                        name=key,
                        system="market",
                        frequency=frequency,
                        source=source,
                        description=instrument.get("name")
                    )

                    rows = write_dataframe(df, indicator=key, system="market")

                    log_fetch(
                        indicator=key,
                        system="market",
                        source=source,
                        rows_fetched=rows,
                        status="success"
                    )

                    logger.info(f"  -> Wrote {rows} rows to database for {key}")

            except Exception as e:
                logger.error(f"  -> Error fetching {key}: {e}")

                if db_enabled:
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

    # ----------------------------------------------------------------------
    # TEST CONNECTION
    # ----------------------------------------------------------------------
    def test_connection(self) -> bool:
        try:
            df = self.fetch_single("SPY")
            return df is not None and not df.empty
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Backward compatibility alias
FetcherYahoo = YahooFetcher


COMMON_TICKERS = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^VIX": "VIX Volatility",

    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLU": "Utilities",

    "TLT": "20+ Year Treasury",
    "IEF": "7-10 Year Treasury",
    "SHY": "1-3 Year Treasury",
    "AGG": "Aggregate Bond",
    "HYG": "High Yield Bond",
    "LQD": "Investment Grade Corporate",

    "GLD": "Gold ETF",
    "SLV": "Silver ETF",
    "USO": "Oil ETF",

    "DX-Y.NYB": "US Dollar Index",
}