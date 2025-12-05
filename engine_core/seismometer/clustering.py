"""
Clustering Drift Detector - K-Means based drift detection

Measures distance from historical cluster centroids to detect regime drift.
"""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging

from .base import BaseDetector

logger = logging.getLogger(__name__)


class ClusteringDriftDetector(BaseDetector):
    """
    K-Means based drift detection.

    Fits K-Means on baseline period, then measures distance from nearest
    centroid for new observations. High distances indicate novel regimes
    not seen during the baseline period.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        lookback_days: int = 252,
        percentile_threshold: float = 95.0,
        random_state: int = 42
    ):
        """
        Initialize clustering drift detector.

        Args:
            n_clusters: Number of K-Means clusters
            lookback_days: Lookback window for calculations
            percentile_threshold: Percentile for normalizing distances
            random_state: Random seed for reproducibility
        """
        super().__init__('clustering_drift', lookback_days)
        self.n_clusters = n_clusters
        self.percentile_threshold = percentile_threshold
        self.random_state = random_state

        self._kmeans: Optional[KMeans] = None
        self._feature_names: Optional[list] = None

    def fit(self, features: pd.DataFrame) -> 'ClusteringDriftDetector':
        """
        Fit K-Means on baseline period.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            self for method chaining
        """
        df = self._validate_features(features)
        self._feature_names = list(df.columns)

        # Get feature matrix and handle NaN
        X = df.values
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]

        if len(X_valid) < self.n_clusters * 10:
            logger.warning(
                f"Limited baseline data: {len(X_valid)} samples for "
                f"{self.n_clusters} clusters"
            )

        # Standardize
        X_scaled = self._standardize(X_valid, fit=True)

        # Fit K-Means
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self._kmeans.fit(X_scaled)

        # Compute distances to nearest centroids for baseline
        distances = self._compute_distances(X_scaled)

        # Store baseline statistics
        self._baseline_stats = {
            'distance_mean': float(np.mean(distances)),
            'distance_std': float(np.std(distances)),
            'distance_p95': float(np.percentile(distances, self.percentile_threshold)),
            'distance_max': float(np.max(distances)),
            'cluster_sizes': np.bincount(self._kmeans.labels_).tolist(),
            'n_samples': len(X_valid),
        }

        self.is_fitted = True
        logger.info(
            f"Fitted {self.name}: {self.n_clusters} clusters, "
            f"baseline p95 distance = {self._baseline_stats['distance_p95']:.4f}"
        )

        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        """
        Score observations by distance from cluster centroids.

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

        # Get feature matrix
        X = df.values

        # Initialize scores with NaN
        scores = np.full(len(df), np.nan)

        # Find valid rows
        valid_mask = ~np.isnan(X).any(axis=1)

        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            X_scaled = self._standardize(X_valid, fit=False)

            # Compute distances
            distances = self._compute_distances(X_scaled)

            # Normalize against baseline p95
            p95 = self._baseline_stats['distance_p95']
            if p95 > 0:
                normalized = distances / p95
            else:
                normalized = distances

            # Clip to [0, 1]
            scores[valid_mask] = self._clip_scores(normalized)

        return pd.Series(scores, index=df.index, name=self.name)

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute distance to nearest centroid for each sample.

        Args:
            X: Scaled feature array (n_samples, n_features)

        Returns:
            Array of distances to nearest centroid
        """
        # Compute distances to all centroids
        centroids = self._kmeans.cluster_centers_
        distances_to_all = np.sqrt(
            ((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2)
        )

        # Return minimum distance (distance to nearest centroid)
        return distances_to_all.min(axis=1)

    def get_cluster_assignments(self, features: pd.DataFrame) -> pd.Series:
        """
        Get cluster assignments for observations.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            Series with cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted - call fit() first")

        df = self._validate_features(features)
        df = df[self._feature_names]

        X = df.values
        assignments = np.full(len(df), -1)

        valid_mask = ~np.isnan(X).any(axis=1)
        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            X_scaled = self._standardize(X_valid, fit=False)
            assignments[valid_mask] = self._kmeans.predict(X_scaled)

        return pd.Series(assignments, index=df.index, name='cluster')
