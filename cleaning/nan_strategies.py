"""
NaN Strategies - Various imputation methods for missing data
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import interpolate
import logging

logger = logging.getLogger(__name__)


class NaNStrategy(ABC):
    """Abstract base class for NaN handling strategies."""

    @abstractmethod
    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        **kwargs
    ) -> pd.Series:
        """
        Fill NaN values in series.

        Args:
            series: Input series with potential NaNs
            max_gap: Maximum consecutive NaNs to fill (None = unlimited)
            **kwargs: Strategy-specific parameters

        Returns:
            Series with NaNs filled
        """
        pass


class ForwardFillStrategy(NaNStrategy):
    """Forward fill - propagate last valid observation."""

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        **kwargs
    ) -> pd.Series:
        result = series.ffill(limit=max_gap)
        return result


class BackwardFillStrategy(NaNStrategy):
    """Backward fill - propagate next valid observation."""

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        **kwargs
    ) -> pd.Series:
        result = series.bfill(limit=max_gap)
        return result


class LinearInterpolateStrategy(NaNStrategy):
    """Linear interpolation between valid points."""

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        **kwargs
    ) -> pd.Series:
        result = series.interpolate(method='linear', limit=max_gap)
        return result


class SplineInterpolateStrategy(NaNStrategy):
    """Spline interpolation for smoother curves."""

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        order: int = 3,
        **kwargs
    ) -> pd.Series:
        # scipy spline needs at least order+1 points
        valid_count = series.notna().sum()
        if valid_count <= order:
            # Fall back to linear
            return series.interpolate(method='linear', limit=max_gap)

        result = series.interpolate(method='spline', order=order, limit=max_gap)
        return result


class MeanFillStrategy(NaNStrategy):
    """Fill with column mean."""

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        **kwargs
    ) -> pd.Series:
        mean_val = series.mean()
        result = series.fillna(mean_val)
        return result


class MedianFillStrategy(NaNStrategy):
    """Fill with column median."""

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        **kwargs
    ) -> pd.Series:
        median_val = series.median()
        result = series.fillna(median_val)
        return result


class ZeroFillStrategy(NaNStrategy):
    """Fill with zero (useful for volume data)."""

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        **kwargs
    ) -> pd.Series:
        result = series.fillna(0)
        return result


class SeasonalDecomposeStrategy(NaNStrategy):
    """
    Fill using seasonal decomposition.
    Useful for data with clear seasonal patterns (e.g., climate data).
    """

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        period: int = 12,
        **kwargs
    ) -> pd.Series:
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Need enough data for decomposition
        if series.notna().sum() < 2 * period:
            return series.interpolate(method='linear', limit=max_gap)

        try:
            # First fill small gaps to enable decomposition
            temp_series = series.interpolate(method='linear', limit=3)

            # Decompose
            decomp = seasonal_decompose(
                temp_series.dropna(),
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )

            # Use seasonal + trend to fill gaps
            reconstructed = decomp.trend + decomp.seasonal
            result = series.fillna(reconstructed)

            return result

        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {e}, using linear")
            return series.interpolate(method='linear', limit=max_gap)


class RollingMeanStrategy(NaNStrategy):
    """Fill with rolling window mean."""

    def fill(
        self,
        series: pd.Series,
        max_gap: Optional[int] = None,
        window: int = 5,
        **kwargs
    ) -> pd.Series:
        rolling_mean = series.rolling(window=window, min_periods=1, center=True).mean()
        result = series.fillna(rolling_mean)
        return result


# Strategy registry
STRATEGIES: Dict[str, NaNStrategy] = {
    "ffill": ForwardFillStrategy(),
    "forward": ForwardFillStrategy(),
    "bfill": BackwardFillStrategy(),
    "backward": BackwardFillStrategy(),
    "linear": LinearInterpolateStrategy(),
    "spline": SplineInterpolateStrategy(),
    "mean": MeanFillStrategy(),
    "median": MedianFillStrategy(),
    "zero": ZeroFillStrategy(),
    "seasonal": SeasonalDecomposeStrategy(),
    "rolling": RollingMeanStrategy(),
}


def get_strategy(name: str) -> NaNStrategy:
    """
    Get a NaN strategy by name.

    Args:
        name: Strategy name ('ffill', 'linear', 'spline', etc.)

    Returns:
        NaNStrategy instance

    Raises:
        ValueError: If strategy name not recognized
    """
    name = name.lower()
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]


def apply_strategy(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    max_gap: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Apply a NaN strategy to a column.

    Args:
        df: Input DataFrame
        column: Column to clean
        strategy: Strategy name
        max_gap: Maximum gap to fill
        **kwargs: Strategy-specific parameters

    Returns:
        DataFrame with cleaned column
    """
    result = df.copy()
    strat = get_strategy(strategy)
    result[column] = strat.fill(df[column], max_gap=max_gap, **kwargs)
    return result
