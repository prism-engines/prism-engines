"""
TDA Lens - Topological Data Analysis

Analyzes the topological structure of indicator relationships.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

from .base_lens import BaseLens


class TDALens(BaseLens):
    """
    TDA Lens: Topological Data Analysis of indicators.

    Provides:
    - Persistent homology analysis
    - Betti numbers (connected components, loops, voids)
    - Topological features for regime detection
    """

    name = "tda"
    description = "Topological data analysis for structural patterns"
    category = "advanced"

    def analyze(
        self,
        df: pd.DataFrame,
        max_dim: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform topological data analysis.

        Args:
            df: Input DataFrame
            max_dim: Maximum homology dimension (0=components, 1=loops)

        Returns:
            Dictionary with TDA results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        # Scale data
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(data[value_cols])

        try:
            # Try to use ripser for persistent homology
            from ripser import ripser
            result_tda = ripser(scaled_values, maxdim=max_dim)
            diagrams = result_tda['dgms']

            # Extract topological features
            topo_features = self._extract_features(diagrams, max_dim)

        except ImportError:
            # Fallback: simplified topological analysis
            topo_features = self._simplified_tda(scaled_values)
            diagrams = None

        # Per-indicator topological contribution
        # (how much does each indicator affect the topology)
        indicator_contribution = self._indicator_contribution(data, value_cols)

        # Rolling topological complexity
        window = kwargs.get("window", 60)
        rolling_complexity = self._rolling_complexity(data, value_cols, window)

        result = {
            "max_dim": max_dim,
            "topological_features": topo_features,
            "indicator_contribution": indicator_contribution,
            "rolling_complexity": rolling_complexity,
            "current_complexity": rolling_complexity[-1] if rolling_complexity else 0,
            "has_ripser": diagrams is not None,
        }

        if diagrams is not None:
            # Add persistence diagram statistics
            for dim in range(max_dim + 1):
                if dim < len(diagrams):
                    dgm = diagrams[dim]
                    finite_dgm = dgm[np.isfinite(dgm[:, 1])]
                    if len(finite_dgm) > 0:
                        lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                        result[f"dim{dim}_count"] = len(finite_dgm)
                        result[f"dim{dim}_max_lifetime"] = float(np.max(lifetimes))
                        result[f"dim{dim}_mean_lifetime"] = float(np.mean(lifetimes))

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def _extract_features(self, diagrams: List, max_dim: int) -> Dict:
        """Extract features from persistence diagrams."""
        features = {}

        for dim in range(max_dim + 1):
            if dim < len(diagrams):
                dgm = diagrams[dim]
                finite_dgm = dgm[np.isfinite(dgm[:, 1])]

                if len(finite_dgm) > 0:
                    lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
                    births = finite_dgm[:, 0]
                    deaths = finite_dgm[:, 1]

                    features[f"betti_{dim}"] = len(finite_dgm)
                    features[f"total_persistence_{dim}"] = float(np.sum(lifetimes))
                    features[f"max_persistence_{dim}"] = float(np.max(lifetimes))
                    features[f"mean_birth_{dim}"] = float(np.mean(births))
                    features[f"entropy_{dim}"] = self._persistence_entropy(lifetimes)
                else:
                    features[f"betti_{dim}"] = 0

        return features

    def _persistence_entropy(self, lifetimes: np.ndarray) -> float:
        """Compute persistence entropy."""
        if len(lifetimes) == 0:
            return 0.0

        total = np.sum(lifetimes)
        if total == 0:
            return 0.0

        probs = lifetimes / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))

        return float(entropy)

    def _simplified_tda(self, data: np.ndarray) -> Dict:
        """Simplified topological analysis without ripser."""
        from scipy.spatial.distance import pdist, squareform

        # Compute distance matrix
        dist_matrix = squareform(pdist(data))

        # Estimate topological complexity via distance statistics
        mean_dist = np.mean(dist_matrix)
        std_dist = np.std(dist_matrix)

        # Approximate Betti-0 via clustering
        from sklearn.cluster import DBSCAN
        eps = mean_dist * 0.5
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            "betti_0_approx": n_clusters,
            "mean_distance": float(mean_dist),
            "distance_std": float(std_dist),
            "complexity_score": float(std_dist / mean_dist) if mean_dist > 0 else 0,
        }

    def _indicator_contribution(
        self,
        data: pd.DataFrame,
        value_cols: List[str]
    ) -> Dict[str, float]:
        """Estimate each indicator's contribution to topological structure."""
        contributions = {}

        # Use variance and correlation as proxies
        for col in value_cols:
            var = data[col].var()
            mean_corr = data[value_cols].corr()[col].abs().mean()
            contributions[col] = float(var * mean_corr)

        # Normalize
        max_contrib = max(contributions.values()) if contributions else 1
        return {k: v / max_contrib for k, v in contributions.items()}

    def _rolling_complexity(
        self,
        data: pd.DataFrame,
        value_cols: List[str],
        window: int
    ) -> List[float]:
        """Compute rolling topological complexity."""
        complexity = []

        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i][value_cols].values

            # Simple complexity measure: mean pairwise distance variance
            from scipy.spatial.distance import pdist
            dists = pdist(window_data)
            comp = float(np.std(dists)) if len(dists) > 0 else 0
            complexity.append(comp)

        return complexity

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by their topological contribution.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        contributions = result["indicator_contribution"]

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in contributions.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)
