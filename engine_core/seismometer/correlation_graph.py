"""
Correlation Graph Detector - Network coherence based instability detection

Measures correlation structure breakdown between features.
Edge weight collapse indicates coherence decay / structural instability.
"""

from typing import Optional
import pandas as pd
import numpy as np
import logging

from .base import BaseDetector

logger = logging.getLogger(__name__)


class CorrelationGraphDetector(BaseDetector):
    """
    Network coherence detector based on correlation structure.

    Computes rolling correlation matrices and measures deviation from
    baseline correlation structure. Structural changes in correlations
    indicate regime instability.

    Key metrics:
    - Average edge strength (mean absolute correlation)
    - Frobenius norm of correlation change
    - Eigenvalue distribution shift
    """

    def __init__(
        self,
        window_days: int = 60,
        lookback_days: int = 252,
        min_periods: int = 30,
        percentile_threshold: float = 95.0
    ):
        """
        Initialize correlation graph detector.

        Args:
            window_days: Rolling window for correlation calculation
            lookback_days: Lookback window for baseline
            min_periods: Minimum observations for correlation
            percentile_threshold: Percentile for normalizing scores
        """
        super().__init__('correlation_graph', lookback_days)
        self.window_days = window_days
        self.min_periods = min_periods
        self.percentile_threshold = percentile_threshold

        self._baseline_corr: Optional[np.ndarray] = None
        self._feature_names: Optional[list] = None

    def fit(self, features: pd.DataFrame) -> 'CorrelationGraphDetector':
        """
        Fit on baseline period - compute reference correlation structure.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            self for method chaining
        """
        df = self._validate_features(features)
        self._feature_names = list(df.columns)

        # Handle NaN by forward-filling
        df = df.ffill().dropna()

        if len(df) < self.min_periods:
            raise ValueError(
                f"Insufficient data: need at least {self.min_periods} rows, "
                f"got {len(df)}"
            )

        # Compute baseline correlation matrix
        self._baseline_corr = df.corr().values

        # Compute rolling correlation distances during baseline
        distances = self._compute_rolling_distances(df)
        valid_distances = distances[~np.isnan(distances)]

        if len(valid_distances) == 0:
            valid_distances = np.array([0.0])

        # Store baseline statistics
        self._baseline_stats = {
            'distance_mean': float(np.mean(valid_distances)),
            'distance_std': float(np.std(valid_distances)),
            'distance_p95': float(np.percentile(valid_distances, self.percentile_threshold)),
            'distance_max': float(np.max(valid_distances)),
            'avg_edge_strength': float(np.mean(np.abs(self._baseline_corr))),
            'n_features': len(self._feature_names),
            'n_samples': len(df),
        }

        self.is_fitted = True
        logger.info(
            f"Fitted {self.name}: {len(self._feature_names)} features, "
            f"baseline avg edge = {self._baseline_stats['avg_edge_strength']:.4f}"
        )

        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        """
        Score observations by correlation structure deviation.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            Series with instability scores in [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted - call fit() first")

        df = self._validate_features(features)

        # Ensure same features
        missing_features = set(self._feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Align columns
        df = df[self._feature_names]

        # Handle NaN by forward-filling
        df = df.ffill()

        # Compute rolling correlation distances
        distances = self._compute_rolling_distances(df)

        # Normalize against baseline p95
        p95 = self._baseline_stats['distance_p95']
        if p95 > 0:
            normalized = distances / p95
        else:
            normalized = distances

        # Clip to [0, 1]
        scores = self._clip_scores(normalized)

        return pd.Series(scores, index=df.index, name=self.name)

    def _compute_rolling_distances(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute rolling correlation distances from baseline.

        Args:
            df: Feature DataFrame

        Returns:
            Array of distances for each time point
        """
        n_samples = len(df)
        distances = np.full(n_samples, np.nan)

        for i in range(self.min_periods, n_samples):
            # Get rolling window
            start_idx = max(0, i - self.window_days)
            window_data = df.iloc[start_idx:i + 1]

            if len(window_data) < self.min_periods:
                continue

            # Compute rolling correlation
            rolling_corr = window_data.corr().values

            # Handle NaN in correlation matrix
            if np.isnan(rolling_corr).any():
                continue

            # Compute distance from baseline
            distances[i] = self._correlation_distance(
                rolling_corr, self._baseline_corr
            )

        return distances

    def _correlation_distance(
        self,
        corr1: np.ndarray,
        corr2: np.ndarray
    ) -> float:
        """
        Compute distance between two correlation matrices.

        Uses normalized Frobenius norm combined with edge strength change.

        Args:
            corr1: First correlation matrix
            corr2: Second correlation matrix (baseline)

        Returns:
            Distance metric (higher = more different)
        """
        # Frobenius norm of difference (normalized by matrix size)
        diff = corr1 - corr2
        n = corr1.shape[0]
        frobenius = np.sqrt(np.sum(diff ** 2)) / n

        # Change in average edge strength
        edge_strength_1 = np.mean(np.abs(corr1))
        edge_strength_2 = np.mean(np.abs(corr2))
        edge_change = abs(edge_strength_1 - edge_strength_2)

        # Combine metrics (weighted average)
        distance = 0.7 * frobenius + 0.3 * edge_change

        return distance

    def get_rolling_correlations(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get rolling correlation statistics over time.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            DataFrame with correlation statistics per date
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted - call fit() first")

        df = self._validate_features(features)
        df = df[self._feature_names].ffill()

        n_samples = len(df)
        stats = []

        for i in range(self.min_periods, n_samples):
            start_idx = max(0, i - self.window_days)
            window_data = df.iloc[start_idx:i + 1]

            if len(window_data) < self.min_periods:
                continue

            rolling_corr = window_data.corr().values

            if np.isnan(rolling_corr).any():
                continue

            # Extract upper triangle (excluding diagonal)
            upper_tri = rolling_corr[np.triu_indices_from(rolling_corr, k=1)]

            stats.append({
                'date': df.index[i],
                'avg_correlation': np.mean(upper_tri),
                'avg_abs_correlation': np.mean(np.abs(upper_tri)),
                'max_correlation': np.max(upper_tri),
                'min_correlation': np.min(upper_tri),
                'correlation_std': np.std(upper_tri),
            })

        return pd.DataFrame(stats).set_index('date')

    def get_correlation_change(
        self,
        features: pd.DataFrame,
        reference_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get pairwise correlation changes from a reference point.

        Args:
            features: DataFrame with date index and feature columns
            reference_date: Date to use as reference (default: use baseline)

        Returns:
            DataFrame with correlation differences
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted - call fit() first")

        df = self._validate_features(features)
        df = df[self._feature_names].ffill()

        if reference_date:
            # Compute correlation up to reference date
            ref_data = df.loc[:reference_date]
            if len(ref_data) < self.min_periods:
                raise ValueError(f"Insufficient data before {reference_date}")
            ref_corr = ref_data.corr()
        else:
            # Use baseline correlation
            ref_corr = pd.DataFrame(
                self._baseline_corr,
                index=self._feature_names,
                columns=self._feature_names
            )

        # Compute current correlation (last window)
        current_corr = df.iloc[-self.window_days:].corr()

        # Compute difference
        corr_change = current_corr - ref_corr

        return corr_change
