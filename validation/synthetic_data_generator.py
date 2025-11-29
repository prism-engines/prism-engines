"""
Synthetic Data Generator - Create test data with known properties
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta


class SyntheticDataGenerator:
    """
    Generate synthetic data with known properties for validation.

    Useful for testing if lenses correctly identify:
    - Known causal relationships
    - Known clusters
    - Known regimes
    - Known anomalies
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_with_known_leader(
        self,
        n_indicators: int = 10,
        n_observations: int = 500,
        leader_idx: int = 0,
        lag: int = 3,
        noise_level: float = 0.1
    ) -> Dict:
        """
        Generate data where one indicator leads others.

        Args:
            n_indicators: Number of indicators
            n_observations: Number of time points
            leader_idx: Index of the leading indicator
            lag: How many periods others lag behind
            noise_level: Amount of noise to add

        Returns:
            Dictionary with data and ground truth
        """
        dates = pd.date_range("2020-01-01", periods=n_observations, freq="D")

        # Generate leader
        leader = np.cumsum(np.random.randn(n_observations))

        # Generate followers with lag
        data = {"date": dates}
        for i in range(n_indicators):
            if i == leader_idx:
                data[f"ind_{i}"] = leader + np.random.randn(n_observations) * noise_level
            else:
                # Lagged version of leader + noise
                lagged = np.zeros(n_observations)
                lagged[lag:] = leader[:-lag]
                lagged[:lag] = leader[0]
                data[f"ind_{i}"] = lagged + np.random.randn(n_observations) * noise_level

        return {
            "data": pd.DataFrame(data),
            "ground_truth": {
                "leader": f"ind_{leader_idx}",
                "lag": lag,
                "followers": [f"ind_{i}" for i in range(n_indicators) if i != leader_idx]
            }
        }

    def generate_with_clusters(
        self,
        n_clusters: int = 3,
        indicators_per_cluster: int = 5,
        n_observations: int = 500,
        within_cluster_corr: float = 0.8
    ) -> Dict:
        """
        Generate data with known indicator clusters.

        Args:
            n_clusters: Number of clusters
            indicators_per_cluster: Indicators per cluster
            n_observations: Number of observations
            within_cluster_corr: Correlation within clusters

        Returns:
            Dictionary with data and ground truth
        """
        dates = pd.date_range("2020-01-01", periods=n_observations, freq="D")
        data = {"date": dates}
        cluster_assignments = {}

        for c in range(n_clusters):
            # Generate cluster base signal
            base = np.cumsum(np.random.randn(n_observations))

            for i in range(indicators_per_cluster):
                idx = c * indicators_per_cluster + i
                # Mix base signal with noise based on within_cluster_corr
                noise = np.random.randn(n_observations)
                signal = within_cluster_corr * base + (1 - within_cluster_corr) * noise
                data[f"ind_{idx}"] = signal
                cluster_assignments[f"ind_{idx}"] = c

        return {
            "data": pd.DataFrame(data),
            "ground_truth": {
                "n_clusters": n_clusters,
                "cluster_assignments": cluster_assignments
            }
        }

    def generate_with_regimes(
        self,
        n_indicators: int = 10,
        n_observations: int = 500,
        regime_lengths: List[int] = [100, 150, 250]
    ) -> Dict:
        """
        Generate data with known regime switches.

        Args:
            n_indicators: Number of indicators
            n_observations: Should match sum of regime_lengths
            regime_lengths: Length of each regime

        Returns:
            Dictionary with data and ground truth
        """
        dates = pd.date_range("2020-01-01", periods=sum(regime_lengths), freq="D")
        data = {"date": dates}

        regime_labels = []
        for i, length in enumerate(regime_lengths):
            regime_labels.extend([i] * length)

        for i in range(n_indicators):
            values = []
            for r, length in enumerate(regime_lengths):
                # Different mean and volatility per regime
                mean = r * 10
                vol = 1 + r * 0.5
                regime_values = np.random.randn(length) * vol + mean
                values.extend(regime_values)
            data[f"ind_{i}"] = values

        return {
            "data": pd.DataFrame(data),
            "ground_truth": {
                "regime_labels": regime_labels,
                "regime_boundaries": np.cumsum(regime_lengths[:-1]).tolist(),
                "n_regimes": len(regime_lengths)
            }
        }
