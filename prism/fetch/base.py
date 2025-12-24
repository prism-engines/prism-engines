"""
PRISM Base Fetcher

Defines the contract all fetchers must implement.

CRITICAL RULES:
1. Fetchers return DataFrames — they NEVER write to the database
2. Fetchers do NOT transform, normalize, or compute derived values
3. Fetchers return raw data exactly as received from source
4. All database writes happen in FetchRunner, not here
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional, List
import pandas as pd
import logging


@dataclass
class FetchResult:
    """
    Result of a single indicator fetch operation.
    
    Attributes:
        indicator_id: The indicator that was fetched
        success: Whether the fetch succeeded
        data: DataFrame with columns [date, value] if successful, None otherwise
        rows: Number of rows fetched (0 if failed)
        first_date: Earliest date in data (None if failed)
        last_date: Latest date in data (None if failed)
        error: Error message if failed, None otherwise
    """
    indicator_id: str
    success: bool
    data: Optional[pd.DataFrame]
    rows: int = 0
    first_date: Optional[date] = None
    last_date: Optional[date] = None
    error: Optional[str] = None
    
    @classmethod
    def from_dataframe(cls, indicator_id: str, df: pd.DataFrame) -> "FetchResult":
        """Create a successful result from a DataFrame."""
        if df is None or df.empty:
            return cls(
                indicator_id=indicator_id,
                success=False,
                data=None,
                error="Empty result"
            )
        
        return cls(
            indicator_id=indicator_id,
            success=True,
            data=df,
            rows=len(df),
            first_date=df["date"].min().date() if hasattr(df["date"].min(), "date") else df["date"].min(),
            last_date=df["date"].max().date() if hasattr(df["date"].max(), "date") else df["date"].max(),
        )
    
    @classmethod
    def from_error(cls, indicator_id: str, error: str) -> "FetchResult":
        """Create a failed result from an error message."""
        return cls(
            indicator_id=indicator_id,
            success=False,
            data=None,
            error=error
        )


class BaseFetcher(ABC):
    """
    Abstract base class for all data fetchers.
    
    Subclasses must implement:
        - fetch(): Fetch a single indicator
        - source_name: Property returning the source identifier
        
    Subclasses must NOT:
        - Write to the database
        - Transform or normalize data
        - Compute derived values
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"prism.fetch.{self.source_name}")
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Return the source identifier (e.g., 'fred', 'tiingo').
        Used for logging and source-to-fetcher mapping.
        """
        pass
    
    @abstractmethod
    def fetch(
        self,
        indicator_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> FetchResult:
        """
        Fetch data for a single indicator.
        
        Args:
            indicator_id: The identifier for the indicator (e.g., 'GDP', 'AAPL')
            start_date: Optional start date for data range
            end_date: Optional end date for data range
            
        Returns:
            FetchResult containing:
                - success: bool
                - data: DataFrame with columns [date, value] or None
                - error: str if failed
                
        The returned DataFrame must have:
            - 'date' column: datetime or date type, sorted ascending
            - 'value' column: numeric type
            
        This method must NOT:
            - Write to any database
            - Transform or normalize values
            - Compute derived indicators
        """
        pass
    
    def fetch_many(
        self,
        indicator_ids: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[FetchResult]:
        """
        Fetch multiple indicators.
        
        Default implementation calls fetch() in sequence.
        Subclasses may override for batch optimization.
        
        Args:
            indicator_ids: List of indicator identifiers
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            List of FetchResult objects, one per indicator
        """
        results = []
        for indicator_id in indicator_ids:
            self.logger.info(f"Fetching {indicator_id}...")
            result = self.fetch(indicator_id, start_date, end_date)
            results.append(result)
            
            if result.success:
                self.logger.info(
                    f"  {indicator_id}: {result.rows} rows "
                    f"({result.first_date} to {result.last_date})"
                )
            else:
                self.logger.warning(f"  {indicator_id}: FAILED - {result.error}")
                
        return results
    
    def validate_dataframe(self, df: pd.DataFrame, indicator_id: str) -> pd.DataFrame:
        """
        Validate and standardize a fetched DataFrame.
        
        Ensures:
            - Required columns exist
            - Date column is proper type
            - Dates are sorted ascending
            - No duplicate dates
            
        Args:
            df: Raw DataFrame from API
            indicator_id: For error messages
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        if df is None or df.empty:
            raise ValueError(f"{indicator_id}: Empty DataFrame")
        
        # Check required columns
        if "date" not in df.columns:
            raise ValueError(f"{indicator_id}: Missing 'date' column")
        if "value" not in df.columns:
            raise ValueError(f"{indicator_id}: Missing 'value' column")
        
        # Ensure date type
        df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date ascending
        df = df.sort_values("date").reset_index(drop=True)
        
        # Remove duplicate dates (keep last)
        df = df.drop_duplicates(subset=["date"], keep="last")
        
        return df
