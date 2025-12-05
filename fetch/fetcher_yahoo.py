"""
Yahoo Finance Fetcher - Stock, ETF, and market data
"""

from pathlib import Path
from typing import Optional, Any, List
import pandas as pd
import logging

from .fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)


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

    # -------------------------------------------------------------
    # UNIFIED FETCH INTERFACE
    # -------------------------------------------------------------
    def fetch_all(
        self,
        registry: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Unified Yahoo Finance fetcher for the entire market registry.

        Args:
            registry: PRISM metric registry dictionary
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Yahoo interval ("1d", "1wk", "1mo")

        Returns:
            DataFrame with merged market data
        """
        if "market" not in registry:
            logger.error("Registry missing 'market' section.")
            return None

        tickers = [item["symbol"] for item in registry["market"]]

        if not tickers:
            logger.error("No Yahoo tickers found in registry['market']")
            return None

        # Attempt batch fetch first (fastest)
        try:
            df = self.fetch_multiple(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Batch fetch failed, falling back to single mode: {e}")

        # Fallback: fetch each individually
        merged = None
        for t in tickers:
            df_t = self.fetch_single(
                t,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            if df_t is None:
                logger.warning(f"Skipping ticker (no data): {t}")
                continue

            if merged is None:
                merged = df_t
            else:
                merged = pd.merge(merged, df_t, on="date", how="outer")

        return merged


def fetch_registry_market_data(
    registry: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch market data for all tickers defined in the registry.

    Args:
        registry: The metric registry dictionary
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with 'date' column and one column per ticker (close prices)
    """
    # Extract tickers from registry
    tickers = []
    for indicator in registry.get("market", []):
        ticker = indicator.get("ticker") or indicator.get("name")
        if ticker:
            tickers.append(ticker)

    if not tickers:
        logger.warning("No market tickers found in registry")
        return None

    logger.info(f"Fetching {len(tickers)} market tickers from Yahoo Finance")

    fetcher = YahooFetcher()

    try:
        df = fetcher.fetch_multiple(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or df.empty:
            logger.warning("No data returned from Yahoo Finance")
            return None

        return df

    except Exception as e:
        logger.error(f"Error fetching registry market data: {e}")
        return None


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
