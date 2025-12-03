"""
Base Fetcher - Abstract interface for all data fetchers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Get directory where this script lives
_SCRIPT_DIR = Path(__file__).parent.resolve()


class BaseFetcher(ABC):
    """
    Abstract base class for all data fetchers.

    All fetchers must implement:
    - fetch_single(): Fetch one ticker/series
    - validate_response(): Validate API response
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize fetcher.

        Args:
            checkpoint_dir: Directory to save fetch checkpoints
        """
        self.checkpoint_dir = checkpoint_dir or (_SCRIPT_DIR / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_fetch_time: Optional[datetime] = None

    @abstractmethod
    def fetch_single(self, ticker: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker/series.

        Args:
            ticker: The ticker symbol or series ID
            **kwargs: Additional parameters (start_date, end_date, etc.)

        Returns:
            DataFrame with 'date' column and ticker data, or None on failure
        """
        pass

    @abstractmethod
    def validate_response(self, response: Any) -> bool:
        """
        Validate the API response before processing.

        Args:
            response: Raw API response

        Returns:
            True if response is valid, False otherwise
        """
        pass

    def fetch_batch(self, tickers: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            **kwargs: Additional parameters passed to fetch_single

        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        results = {}

        for ticker in tickers:
            logger.info(f"Fetching {ticker}...")
            try:
                df = self.fetch_single(ticker, **kwargs)
                if df is not None and not df.empty:
                    results[ticker] = df
                    logger.info(f"  -> {len(df)} rows fetched")
                else:
                    logger.warning(f"  -> No data returned for {ticker}")
            except Exception as e:
                logger.error(f"  -> Error fetching {ticker}: {e}")

        return results

    def sanitize_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Standardize DataFrame format.

        - Ensures 'date' column exists
        - Flattens MultiIndex columns
        - Lowercases column names

        Args:
            df: Raw DataFrame
            ticker: Ticker name for column naming

        Returns:
            Sanitized DataFrame
        """
        if df is None or df.empty:
            logger.warning(f"{ticker} returned empty dataset")
            return pd.DataFrame({"date": [], ticker.lower(): []})

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(x) for x in col if x)
                for col in df.columns
            ]

        # Lowercase all column names
        df.columns = [c.lower() for c in df.columns]

        # Ensure date column exists
        if "date" not in df.columns:
            if df.index.name and "date" in df.index.name.lower():
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
            else:
                raise ValueError(f"'{ticker}' missing date column")

        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])

        return df

    def save_checkpoint(self, data: Dict[str, pd.DataFrame], name: str) -> Path:
        """
        Save fetch results as checkpoint.

        Args:
            data: Dictionary of DataFrames
            name: Checkpoint name

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"{name}_{timestamp}.parquet"

        # Combine all DataFrames
        if data:
            combined = pd.concat(
                [df.assign(source_ticker=ticker) for ticker, df in data.items()],
                ignore_index=True
            )
            combined.to_parquet(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def get_fetch_summary(self, results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate summary of fetch results.

        Args:
            results: Dictionary of fetched DataFrames

        Returns:
            Summary dictionary
        """
        summary = {
            "n_tickers": len(results),
            "tickers": list(results.keys()),
            "total_rows": sum(len(df) for df in results.values()),
            "fetch_time": datetime.now().isoformat(),
            "date_ranges": {}
        }

        for ticker, df in results.items():
            if "date" in df.columns and len(df) > 0:
                summary["date_ranges"][ticker] = {
                    "start": df["date"].min().isoformat(),
                    "end": df["date"].max().isoformat(),
                    "rows": len(df)
                }

        return summary
