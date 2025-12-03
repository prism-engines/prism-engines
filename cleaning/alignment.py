"""
Data Alignment - Align data of different frequencies
"""

from typing import Dict, List, Optional, Literal
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

FrequencyType = Literal["D", "W", "M", "Q", "Y"]


class DataAligner:
    """
    Align data from different sources to a common frequency.

    Handles:
    - Upsampling (e.g., monthly -> daily)
    - Downsampling (e.g., daily -> weekly)
    - Date range alignment
    """

    # Pandas frequency codes
    FREQ_MAP = {
        "D": "D",      # Daily
        "W": "W-FRI",  # Weekly (Friday close)
        "M": "ME",     # Month end
        "Q": "QE",     # Quarter end
        "Y": "YE",     # Year end
        "daily": "D",
        "weekly": "W-FRI",
        "monthly": "ME",
        "quarterly": "QE",
        "yearly": "YE",
    }

    def __init__(self, date_col: str = "date"):
        """
        Initialize aligner.

        Args:
            date_col: Name of date column in DataFrames
        """
        self.date_col = date_col

    def align_to_frequency(
        self,
        df: pd.DataFrame,
        target_freq: FrequencyType,
        agg_method: str = "last",
        fill_method: Optional[str] = "ffill"
    ) -> pd.DataFrame:
        """
        Align DataFrame to target frequency.

        Args:
            df: Input DataFrame with date column
            target_freq: Target frequency ('D', 'W', 'M', 'Q', 'Y')
            agg_method: Aggregation for downsampling ('last', 'mean', 'sum', 'first')
            fill_method: Fill method for upsampling ('ffill', 'bfill', None)

        Returns:
            Aligned DataFrame
        """
        result = df.copy()

        # Ensure date column is datetime and set as index
        result[self.date_col] = pd.to_datetime(result[self.date_col])
        result = result.set_index(self.date_col)

        # Get pandas frequency string
        freq = self.FREQ_MAP.get(target_freq, target_freq)

        # Detect current frequency
        inferred_freq = pd.infer_freq(result.index)

        # Resample
        if agg_method == "last":
            result = result.resample(freq).last()
        elif agg_method == "first":
            result = result.resample(freq).first()
        elif agg_method == "mean":
            result = result.resample(freq).mean()
        elif agg_method == "sum":
            result = result.resample(freq).sum()
        else:
            result = result.resample(freq).last()

        # Fill if upsampling created gaps
        if fill_method == "ffill":
            result = result.ffill()
        elif fill_method == "bfill":
            result = result.bfill()

        # Reset index
        result = result.reset_index()

        return result

    def align_multiple(
        self,
        dfs: Dict[str, pd.DataFrame],
        target_freq: FrequencyType,
        agg_method: str = "last"
    ) -> pd.DataFrame:
        """
        Align multiple DataFrames to same frequency and merge.

        Args:
            dfs: Dictionary of {name: DataFrame}
            target_freq: Target frequency
            agg_method: Aggregation method

        Returns:
            Merged DataFrame with all columns aligned
        """
        aligned_dfs = []

        for name, df in dfs.items():
            aligned = self.align_to_frequency(df, target_freq, agg_method)
            aligned_dfs.append(aligned)

        # Merge all on date
        if not aligned_dfs:
            return pd.DataFrame()

        result = aligned_dfs[0]
        for df in aligned_dfs[1:]:
            result = result.merge(df, on=self.date_col, how="outer")

        result = result.sort_values(self.date_col)
        return result

    def align_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        method: str = "inner"
    ) -> pd.DataFrame:
        """
        Align DataFrame to specific date range.

        Args:
            df: Input DataFrame
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            method: 'inner' (only where data exists) or 'outer' (fill to range)

        Returns:
            DataFrame filtered/extended to date range
        """
        result = df.copy()
        result[self.date_col] = pd.to_datetime(result[self.date_col])

        if start_date:
            start = pd.to_datetime(start_date)
            if method == "inner":
                result = result[result[self.date_col] >= start]
            # outer would need to extend index

        if end_date:
            end = pd.to_datetime(end_date)
            if method == "inner":
                result = result[result[self.date_col] <= end]

        return result

    def get_common_date_range(
        self,
        dfs: List[pd.DataFrame]
    ) -> tuple:
        """
        Find common date range across multiple DataFrames.

        Args:
            dfs: List of DataFrames

        Returns:
            Tuple of (start_date, end_date)
        """
        starts = []
        ends = []

        for df in dfs:
            if self.date_col in df.columns and len(df) > 0:
                dates = pd.to_datetime(df[self.date_col])
                starts.append(dates.min())
                ends.append(dates.max())

        if not starts:
            return None, None

        common_start = max(starts)
        common_end = min(ends)

        return common_start, common_end

    def create_master_panel(
        self,
        dfs: Dict[str, pd.DataFrame],
        target_freq: FrequencyType = "D",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agg_method: str = "last"
    ) -> pd.DataFrame:
        """
        Create master panel from multiple data sources.

        Args:
            dfs: Dictionary of {name: DataFrame}
            target_freq: Target frequency
            start_date: Start date filter
            end_date: End date filter
            agg_method: Aggregation method

        Returns:
            Master panel DataFrame
        """
        # Align all to target frequency
        panel = self.align_multiple(dfs, target_freq, agg_method)

        # Filter date range
        if start_date or end_date:
            panel = self.align_date_range(panel, start_date, end_date)

        # Sort by date
        panel = panel.sort_values(self.date_col).reset_index(drop=True)

        logger.info(
            f"Created master panel: {len(panel)} rows, "
            f"{len(panel.columns) - 1} indicators"
        )

        return panel
