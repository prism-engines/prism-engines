"""
PRISM Distance Engine

Computes geometric distances between indicators.

Measures:
- Euclidean distance
- Mahalanobis distance (accounts for covariance)
- Cosine distance (directional similarity)

Phase: Unbound
Normalization: Z-score preferred
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from .base import BaseEngine


logger = logging.getLogger(__name__)


def _euclidean_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Treats each indicator as a vector in time-space.
    """
    X = df.values.T  # (n_indicators, n_samples)
    n = X.shape[0]

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(X[i] - X[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def _mahalanobis_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise Mahalanobis distances.

    Accounts for covariance structure of the data.
    Uses time-wise covariance (treating indicators as features at each time point).
    """
    X = df.values  # (n_samples, n_indicators)
    n_indicators = X.shape[1]

    # Compute covariance matrix across indicators (correlation structure)
    cov = np.cov(X.T)  # (n_indicators, n_indicators)

    # Handle 1D case
    if cov.ndim == 0:
        cov = np.array([[cov]])

    # Regularize if singular
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Regularize
        cov_inv = np.linalg.inv(cov + 0.01 * np.eye(n_indicators))

    # Compute mean vectors for each indicator (its time series)
    # Then compute Mahalanobis distance between mean-centered representations
    means = X.mean(axis=0)  # (n_indicators,)

    distances = np.zeros((n_indicators, n_indicators))
    for i in range(n_indicators):
        for j in range(i + 1, n_indicators):
            # Use the difference in loadings/correlation space
            diff = np.zeros(n_indicators)
            diff[i] = 1
            diff[j] = -1
            dist = np.sqrt(np.abs(diff @ cov_inv @ diff))
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def _cosine_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise cosine distances.

    Measures directional similarity (1 - cosine similarity).
    """
    X = df.values.T  # (n_indicators, n_samples)
    n = X.shape[0]

    # Normalize to unit vectors
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    X_norm = X / norms

    # Cosine similarity
    similarity = X_norm @ X_norm.T

    # Convert to distance
    distances = 1 - similarity
    np.fill_diagonal(distances, 0)

    return distances


def _correlation_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute correlation distance.

    Distance = 1 - |correlation|
    """
    corr = df.corr().values
    distances = 1 - np.abs(corr)
    np.fill_diagonal(distances, 0)

    return distances


class DistanceEngine(BaseEngine):
    """
    Distance engine.

    Computes multiple geometric distance metrics between indicators.

    Outputs:
        - derived.distances: Pairwise distance matrices
    """

    name = "distance"
    phase = "derived"
    default_normalization = "zscore"

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        methods: list = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run distance analysis.

        Args:
            df: Normalized indicator data
            run_id: Unique run identifier
            methods: List of distance methods (default: all)

        Returns:
            Dict with summary metrics
        """
        if methods is None:
            methods = ["euclidean", "mahalanobis", "cosine", "correlation"]

        df_clean = df
        indicators = df_clean.columns.tolist()
        n_indicators = len(indicators)

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Compute distance matrices
        distance_matrices = {}

        if "euclidean" in methods:
            distance_matrices["euclidean"] = _euclidean_distance_matrix(df_clean)

        if "mahalanobis" in methods:
            try:
                distance_matrices["mahalanobis"] = _mahalanobis_distance_matrix(df_clean)
            except Exception as e:
                logger.warning(f"Mahalanobis distance failed: {e}")

        if "cosine" in methods:
            distance_matrices["cosine"] = _cosine_distance_matrix(df_clean)

        if "correlation" in methods:
            distance_matrices["correlation"] = _correlation_distance_matrix(df_clean)

        # Store results
        records = []
        for i in range(n_indicators):
            for j in range(i + 1, n_indicators):
                record = {
                    "indicator_1": indicators[i],
                    "indicator_2": indicators[j],
                    "window_start": window_start,
                    "window_end": window_end,
                    "run_id": run_id,
                }

                for method, matrix in distance_matrices.items():
                    record[f"{method}_distance"] = float(matrix[i, j])

                records.append(record)

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("distances", df_results, run_id)

        # Summary metrics
        metrics = {
            "n_indicators": n_indicators,
            "n_pairs": len(records),
            "n_samples": len(df_clean),
            "methods": methods,
        }

        for method, matrix in distance_matrices.items():
            condensed = squareform(matrix)
            metrics[f"avg_{method}_distance"] = float(np.mean(condensed))
            metrics[f"max_{method}_distance"] = float(np.max(condensed))
            metrics[f"min_{method}_distance"] = float(np.min(condensed))

        logger.info(
            f"Distance complete: {n_indicators} indicators, "
            f"methods={methods}"
        )

        return metrics
