"""
NaN Analyzer - Analyze missing data patterns before cleaning
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NaNAnalyzer:
    """
    Analyze NaN patterns in data to inform cleaning strategy.

    Key analyses:
    - Missing data percentage per column
    - Gap patterns (consecutive NaNs)
    - Temporal patterns (are NaNs clustered in time?)
    - Cross-column correlation of missingness
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "date"):
        """
        Initialize analyzer.

        Args:
            df: DataFrame to analyze
            date_col: Name of date column
        """
        self.df = df.copy()
        self.date_col = date_col

        # Ensure date is datetime and sorted
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col)

        self.value_cols = [c for c in self.df.columns if c != date_col]

    def full_report(self) -> Dict:
        """Generate comprehensive NaN analysis report."""
        return {
            "summary": self.summary(),
            "gap_analysis": self.gap_analysis(),
            "temporal_pattern": self.temporal_pattern(),
            "recommendations": self.recommend_strategies()
        }

    def summary(self) -> Dict[str, Dict]:
        """
        Basic summary of missing values per column.

        Returns:
            Dictionary with stats per column
        """
        summary = {}
        n_total = len(self.df)

        for col in self.value_cols:
            n_nan = self.df[col].isna().sum()
            n_valid = n_total - n_nan

            # Date range of valid data
            valid_data = self.df[self.df[col].notna()]
            if len(valid_data) > 0 and self.date_col in self.df.columns:
                first_valid = valid_data[self.date_col].min()
                last_valid = valid_data[self.date_col].max()
            else:
                first_valid = last_valid = None

            summary[col] = {
                "n_total": n_total,
                "n_nan": int(n_nan),
                "n_valid": int(n_valid),
                "nan_pct": round(n_nan / n_total * 100, 2) if n_total > 0 else 0,
                "first_valid_date": str(first_valid) if first_valid else None,
                "last_valid_date": str(last_valid) if last_valid else None,
            }

        return summary

    def gap_analysis(self) -> Dict[str, Dict]:
        """
        Analyze gaps (consecutive NaN sequences).

        Returns:
            Dictionary with gap statistics per column
        """
        gap_stats = {}

        for col in self.value_cols:
            is_nan = self.df[col].isna()

            # Find gap lengths
            gaps = []
            current_gap = 0

            for val in is_nan:
                if val:
                    current_gap += 1
                elif current_gap > 0:
                    gaps.append(current_gap)
                    current_gap = 0
            if current_gap > 0:
                gaps.append(current_gap)

            if gaps:
                gap_stats[col] = {
                    "n_gaps": len(gaps),
                    "max_gap": max(gaps),
                    "mean_gap": round(np.mean(gaps), 1),
                    "median_gap": int(np.median(gaps)),
                    "total_missing_in_gaps": sum(gaps),
                }
            else:
                gap_stats[col] = {
                    "n_gaps": 0,
                    "max_gap": 0,
                    "mean_gap": 0,
                    "median_gap": 0,
                    "total_missing_in_gaps": 0,
                }

        return gap_stats

    def temporal_pattern(self) -> Dict[str, Dict]:
        """
        Analyze if NaNs are clustered in specific time periods.

        Returns:
            Dictionary with temporal patterns
        """
        if self.date_col not in self.df.columns:
            return {}

        temporal = {}

        for col in self.value_cols:
            # Count NaNs by year
            df_temp = self.df[[self.date_col, col]].copy()
            df_temp["year"] = df_temp[self.date_col].dt.year

            yearly_nan = df_temp.groupby("year")[col].apply(
                lambda x: x.isna().sum()
            ).to_dict()

            yearly_total = df_temp.groupby("year")[col].count().to_dict()

            # Find worst years (highest NaN %)
            nan_pct_by_year = {}
            for year in yearly_nan:
                total = yearly_total.get(year, 0) + yearly_nan.get(year, 0)
                if total > 0:
                    nan_pct_by_year[year] = round(yearly_nan[year] / total * 100, 1)

            # Top 3 worst years
            worst_years = sorted(nan_pct_by_year.items(), key=lambda x: -x[1])[:3]

            temporal[col] = {
                "nan_by_year": yearly_nan,
                "worst_years": worst_years,
            }

        return temporal

    def recommend_strategies(self) -> Dict[str, str]:
        """
        Recommend cleaning strategy based on gap analysis.

        Returns:
            Dictionary mapping column -> recommended strategy
        """
        recommendations = {}
        gap_stats = self.gap_analysis()
        summary = self.summary()

        for col in self.value_cols:
            gaps = gap_stats.get(col, {})
            stats = summary.get(col, {})

            max_gap = gaps.get("max_gap", 0)
            nan_pct = stats.get("nan_pct", 0)

            # Decision logic
            if nan_pct == 0:
                recommendations[col] = "none"  # No cleaning needed
            elif nan_pct > 50:
                recommendations[col] = "drop_column"  # Too much missing
            elif max_gap <= 3:
                recommendations[col] = "ffill"  # Small gaps, forward fill ok
            elif max_gap <= 10:
                recommendations[col] = "linear"  # Medium gaps, interpolate
            elif max_gap <= 30:
                recommendations[col] = "spline"  # Larger gaps, smooth interpolation
            else:
                recommendations[col] = "segment"  # Very large gaps, handle separately

        return recommendations

    def missing_correlation(self) -> pd.DataFrame:
        """
        Analyze if missingness in different columns is correlated.

        Returns:
            Correlation matrix of missing patterns
        """
        # Create binary matrix: 1 = missing, 0 = present
        missing_matrix = self.df[self.value_cols].isna().astype(int)

        # Compute correlation
        return missing_matrix.corr()
