"""
Base Detector - Abstract interface for all instability detectors
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Abstract base class for all instability detectors.

    All detectors follow the same pattern:
    - fit() on baseline period (normal market behavior)
    - score() new observations returning 0-1 (higher = more unstable)
    """

    def __init__(self, name: str, lookback_days: int = 252):
        """
        Initialize detector.

        Args:
            name: Detector name for identification
            lookback_days: Number of days to use for rolling calculations
        """
        self.name = name
        self.lookback_days = lookback_days
        self.is_fitted = False
        self._baseline_stats: Dict[str, Any] = {}
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> 'BaseDetector':
        """
        Fit on baseline period (normal market behavior).

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def score(self, features: pd.DataFrame) -> pd.Series:
        """
        Return instability scores 0-1, indexed by date.

        Higher scores indicate more instability/anomaly.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            Series with date index and scores in [0, 1]
        """
        pass

    def fit_score(self, features: pd.DataFrame, baseline_end: str) -> pd.Series:
        """
        Convenience method: fit on baseline, score everything.

        Args:
            features: Full DataFrame with date index
            baseline_end: End date for baseline period (e.g., '2019-12-31')

        Returns:
            Series with instability scores for all dates
        """
        self.fit(features.loc[:baseline_end])
        return self.score(features)

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare features DataFrame.

        Args:
            features: Input DataFrame

        Returns:
            Cleaned DataFrame with DatetimeIndex

        Raises:
            ValueError: If validation fails
        """
        if features is None or features.empty:
            raise ValueError("Features DataFrame is empty")

        # Ensure we have a copy
        df = features.copy()

        # Handle date index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Check for sufficient data
        if len(df) < 10:
            raise ValueError(f"Insufficient data: need at least 10 rows, got {len(df)}")

        return df

    def _standardize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Standardize features to zero mean and unit variance.

        Args:
            X: Feature array (n_samples, n_features)
            fit: If True, compute and store mean/std from this data

        Returns:
            Standardized array
        """
        if fit:
            self._scaler_mean = np.nanmean(X, axis=0)
            self._scaler_std = np.nanstd(X, axis=0)
            # Prevent division by zero
            self._scaler_std[self._scaler_std < 1e-8] = 1.0

        if self._scaler_mean is None or self._scaler_std is None:
            raise ValueError("Scaler not fitted - call with fit=True first")

        return (X - self._scaler_mean) / self._scaler_std

    def _clip_scores(self, scores: np.ndarray) -> np.ndarray:
        """Clip scores to [0, 1] range."""
        return np.clip(scores, 0.0, 1.0)

    def get_baseline_stats(self) -> Dict[str, Any]:
        """Get stored baseline statistics."""
        return self._baseline_stats.copy()

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {fitted_str})"
