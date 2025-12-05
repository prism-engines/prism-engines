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


# =============================================================================
# REGISTRY-DRIVEN TICKER MAPPING
# =============================================================================

# Mapping from lowercase registry names to Yahoo Finance symbols
# Registry stores lowercase, Yahoo API needs proper symbols
REGISTRY_TO_YAHOO_MAP = {
    # Major ETFs/Indices (direct uppercase mapping)
    "spy": "SPY",
    "qqq": "QQQ",
    "iwm": "IWM",
    "gld": "GLD",
    "tlt": "TLT",
    "btc": "BTC-USD",  # Bitcoin needs -USD suffix

    # Special symbols
    "dxy": "DX-Y.NYB",  # US Dollar Index
    "vix": "^VIX",      # VIX needs caret prefix

    # Sector ETFs (direct uppercase)
    "xlc": "XLC",
    "xly": "XLY",
    "xlp": "XLP",
    "xle": "XLE",
    "xlf": "XLF",
    "xlv": "XLV",
    "xli": "XLI",
    "xlb": "XLB",
    "xlre": "XLRE",
    "xlk": "XLK",
    "xlu": "XLU",
}


def get_yahoo_tickers_from_registry(reg: dict) -> list[str]:
    """
    Get list of Yahoo Finance tickers from the registry's market section.

    Maps lowercase registry names to proper Yahoo Finance symbols.
    DB stores lowercase names matching registry, not Yahoo symbols.

    Args:
        reg: The metric registry dictionary.

    Returns:
        List of Yahoo Finance ticker symbols (uppercase/proper format).
    """
    market_metrics = reg.get("market", [])
    yahoo_tickers = []

    for metric in market_metrics:
        name = metric.get("name", "").lower()
        metric_type = metric.get("type", "")

        # Only process market_price types
        if metric_type != "market_price":
            continue

        # Use mapping if available, otherwise uppercase the name
        yahoo_symbol = REGISTRY_TO_YAHOO_MAP.get(name, name.upper())
        yahoo_tickers.append(yahoo_symbol)

    return yahoo_tickers


def get_registry_name_from_yahoo(yahoo_ticker: str) -> str:
    """
    Convert Yahoo Finance ticker back to lowercase registry name.

    Args:
        yahoo_ticker: Yahoo Finance ticker symbol.

    Returns:
        Lowercase registry name for database storage.
    """
    # Build reverse mapping
    yahoo_to_registry = {v: k for k, v in REGISTRY_TO_YAHOO_MAP.items()}

    if yahoo_ticker in yahoo_to_registry:
        return yahoo_to_registry[yahoo_ticker]

    # Default: lowercase the ticker
    return yahoo_ticker.lower().replace("-usd", "").replace("^", "")


def fetch_registry_market_data(
    reg: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch all market data defined in the registry.

    Returns DataFrame with lowercase column names matching registry names.

    Args:
        reg: The metric registry dictionary.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        DataFrame with date column and lowercase registry names as columns.
    """
    import yfinance as yf

    yahoo_tickers = get_yahoo_tickers_from_registry(reg)

    if not yahoo_tickers:
        logger.warning("No Yahoo tickers found in registry")
        return pd.DataFrame()

    try:
        # Fetch all tickers at once
        df = yf.download(
            yahoo_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            logger.warning("No data returned from Yahoo Finance")
            return pd.DataFrame()

        # Handle MultiIndex columns from multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]

        df = df.reset_index()

        # Rename columns to lowercase registry names
        new_columns = ["date"]
        for ticker in yahoo_tickers:
            registry_name = get_registry_name_from_yahoo(ticker)
            new_columns.append(registry_name)

        df.columns = new_columns

        return df

    except Exception as e:
        logger.error(f"Error fetching registry market data: {e}")
        return pd.DataFrame()
