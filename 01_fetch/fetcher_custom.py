"""
Custom Fetcher - Load data from local files (CSV, Excel, Parquet)
"""

from pathlib import Path
from typing import Optional, Any, Union
import pandas as pd
import logging

from .fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)


class CustomFetcher(BaseFetcher):
    """
    Fetcher for custom/local data files.

    Supports CSV, Excel, and Parquet formats.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize Custom fetcher.

        Args:
            data_dir: Directory containing data files
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__(checkpoint_dir)
        self.data_dir = Path(data_dir) if data_dir else Path(".")

    def validate_response(self, response: Any) -> bool:
        """Validate loaded data."""
        if response is None:
            return False
        if isinstance(response, pd.DataFrame) and response.empty:
            return False
        return True

    def fetch_single(
        self,
        ticker: str,
        file_path: Optional[Union[str, Path]] = None,
        date_column: str = "date",
        value_column: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Load data from a local file.

        Args:
            ticker: Name for the data series
            file_path: Path to file (or searches in data_dir for ticker.csv)
            date_column: Name of date column in source file
            value_column: Name of value column (defaults to ticker name)
            **kwargs: Additional arguments passed to pandas read functions

        Returns:
            DataFrame with date and value columns
        """
        # Find file path
        if file_path is None:
            # Search for file matching ticker name
            for ext in [".csv", ".parquet", ".xlsx", ".xls"]:
                candidate = self.data_dir / f"{ticker}{ext}"
                if candidate.exists():
                    file_path = candidate
                    break

        if file_path is None:
            logger.error(f"No file found for {ticker} in {self.data_dir}")
            return None

        file_path = Path(file_path)

        try:
            # Load based on file extension
            df = self._load_file(file_path, **kwargs)

            if not self.validate_response(df):
                return None

            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            date_column = date_column.lower()

            # Find and rename date column
            if date_column in df.columns:
                df = df.rename(columns={date_column: "date"})
            elif "date" not in df.columns:
                # Try to find a date-like column
                date_cols = [c for c in df.columns if "date" in c or "time" in c]
                if date_cols:
                    df = df.rename(columns={date_cols[0]: "date"})
                else:
                    logger.error(f"No date column found in {file_path}")
                    return None

            # Handle value column
            if value_column:
                value_column = value_column.lower()
                if value_column in df.columns:
                    df = df.rename(columns={value_column: ticker.lower()})

            return self.sanitize_dataframe(df, ticker)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _load_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load file based on extension.

        Args:
            file_path: Path to file
            **kwargs: Additional pandas arguments

        Returns:
            DataFrame
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(file_path, **kwargs)
        elif suffix == ".parquet":
            return pd.read_parquet(file_path, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path, **kwargs)
        elif suffix == ".json":
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def load_panel(
        self,
        file_path: Union[str, Path],
        date_column: str = "date",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a pre-built panel (multiple series in one file).

        Args:
            file_path: Path to panel file
            date_column: Name of date column
            **kwargs: Additional pandas arguments

        Returns:
            DataFrame with date and multiple value columns
        """
        file_path = Path(file_path)

        try:
            df = self._load_file(file_path, **kwargs)

            # Standardize
            df.columns = [c.lower() for c in df.columns]

            if date_column.lower() != "date" and date_column.lower() in df.columns:
                df = df.rename(columns={date_column.lower(): "date"})

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            return df

        except Exception as e:
            logger.error(f"Error loading panel from {file_path}: {e}")
            return pd.DataFrame()
