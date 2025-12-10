"""
Outlier Detection - Flag suspicious values in data
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Detect outliers using multiple methods.

    Methods:
    - Z-score: Flag values > N standard deviations from mean
    - IQR: Flag values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR
    - MAD: Median Absolute Deviation (robust to outliers)
    - Rolling: Compare to local rolling statistics
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "date"):
        """
        Initialize detector.

        Args:
            df: DataFrame to analyze
            date_col: Name of date column
        """
        self.df = df.copy()
        self.date_col = date_col
        self.value_cols = [c for c in df.columns if c != date_col]

    def detect_zscore(
        self,
        threshold: float = 3.0,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        Detect outliers using Z-score method.

        Args:
            threshold: Number of standard deviations for outlier
            columns: Columns to check (default: all value columns)

        Returns:
            Dictionary mapping column -> boolean Series (True = outlier)
        """
        columns = columns or self.value_cols
        outliers = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            series = self.df[col].dropna()
            if len(series) == 0:
                outliers[col] = pd.Series(False, index=self.df.index)
                continue

            z_scores = np.abs(stats.zscore(series))
            outlier_mask = pd.Series(False, index=self.df.index)
            outlier_mask.loc[series.index] = z_scores > threshold
            outliers[col] = outlier_mask

        return outliers

    def detect_iqr(
        self,
        multiplier: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        Detect outliers using IQR method.

        Args:
            multiplier: IQR multiplier for bounds (1.5 = standard, 3.0 = extreme)
            columns: Columns to check

        Returns:
            Dictionary mapping column -> boolean Series (True = outlier)
        """
        columns = columns or self.value_cols
        outliers = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            series = self.df[col]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outliers[col] = (series < lower_bound) | (series > upper_bound)

        return outliers

    def detect_mad(
        self,
        threshold: float = 3.0,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        Detect outliers using Median Absolute Deviation (robust method).

        Args:
            threshold: Number of MADs for outlier
            columns: Columns to check

        Returns:
            Dictionary mapping column -> boolean Series (True = outlier)
        """
        columns = columns or self.value_cols
        outliers = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            series = self.df[col]
            median = series.median()
            mad = np.median(np.abs(series - median))

            if mad == 0:
                # All values are the same
                outliers[col] = pd.Series(False, index=self.df.index)
                continue

            # Modified Z-score using MAD
            modified_z = 0.6745 * (series - median) / mad
            outliers[col] = np.abs(modified_z) > threshold

        return outliers

    def detect_rolling(
        self,
        window: int = 20,
        std_threshold: float = 3.0,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        Detect outliers compared to local rolling statistics.

        Useful for detecting sudden jumps in time series.

        Args:
            window: Rolling window size
            std_threshold: Number of rolling stds for outlier
            columns: Columns to check

        Returns:
            Dictionary mapping column -> boolean Series (True = outlier)
        """
        columns = columns or self.value_cols
        outliers = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            series = self.df[col]
            rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
            rolling_std = series.rolling(window=window, center=True, min_periods=1).std()

            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan)

            deviation = np.abs(series - rolling_mean) / rolling_std
            outliers[col] = deviation > std_threshold

        return outliers

    def detect_all(
        self,
        columns: Optional[List[str]] = None,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        mad_threshold: float = 3.0
    ) -> Dict[str, Dict[str, pd.Series]]:
        """
        Run all detection methods.

        Args:
            columns: Columns to check
            z_threshold: Z-score threshold
            iqr_multiplier: IQR multiplier
            mad_threshold: MAD threshold

        Returns:
            Nested dict: {method: {column: outlier_mask}}
        """
        return {
            "zscore": self.detect_zscore(z_threshold, columns),
            "iqr": self.detect_iqr(iqr_multiplier, columns),
            "mad": self.detect_mad(mad_threshold, columns),
            "rolling": self.detect_rolling(columns=columns),
        }

    def consensus_outliers(
        self,
        columns: Optional[List[str]] = None,
        min_methods: int = 2
    ) -> Dict[str, pd.Series]:
        """
        Find outliers flagged by multiple methods.

        Args:
            columns: Columns to check
            min_methods: Minimum number of methods that must agree

        Returns:
            Dictionary mapping column -> boolean Series
        """
        all_results = self.detect_all(columns)
        columns = columns or self.value_cols
        consensus = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            # Count how many methods flag each point
            method_count = pd.Series(0, index=self.df.index)
            for method_results in all_results.values():
                if col in method_results:
                    method_count += method_results[col].astype(int)

            consensus[col] = method_count >= min_methods

        return consensus

    def summary_report(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate summary report of outliers per column.

        Args:
            columns: Columns to analyze

        Returns:
            DataFrame with outlier counts per method per column
        """
        all_results = self.detect_all(columns)
        columns = columns or self.value_cols

        report_data = []
        for col in columns:
            row = {"column": col, "n_total": len(self.df)}
            for method, results in all_results.items():
                if col in results:
                    row[f"{method}_count"] = results[col].sum()
                else:
                    row[f"{method}_count"] = 0
            report_data.append(row)

        return pd.DataFrame(report_data)
