"""
Transfer Entropy Lens - Information flow analysis

Measures directed information transfer between indicators.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

from .base_lens import BaseLens


class TransferEntropyLens(BaseLens):
    """
    Transfer Entropy Lens: Measure information flow between indicators.

    Provides:
    - Pairwise transfer entropy
    - Net information flow
    - Information hubs and sinks
    """

    name = "transfer_entropy"
    description = "Transfer entropy for directed information flow"
    category = "advanced"

    def analyze(
        self,
        df: pd.DataFrame,
        lag: int = 1,
        k: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute transfer entropy between indicators.

        Args:
            df: Input DataFrame
            lag: Time lag for transfer entropy
            k: Number of neighbors for entropy estimation

        Returns:
            Dictionary with transfer entropy results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        # Limit columns for computational efficiency
        max_cols = kwargs.get("max_columns", 15)
        if len(value_cols) > max_cols:
            variances = data[value_cols].var().sort_values(ascending=False)
            value_cols = variances.head(max_cols).index.tolist()

        # Compute pairwise transfer entropy
        te_matrix = pd.DataFrame(
            np.zeros((len(value_cols), len(value_cols))),
            index=value_cols,
            columns=value_cols
        )

        for source in value_cols:
            for target in value_cols:
                if source == target:
                    continue

                te = self._transfer_entropy(
                    data[source].values,
                    data[target].values,
                    lag=lag,
                    k=k
                )
                te_matrix.loc[source, target] = te

        # Net transfer (outflow - inflow)
        outflow = te_matrix.sum(axis=1)
        inflow = te_matrix.sum(axis=0)
        net_transfer = (outflow - inflow).to_dict()

        # Information hubs (high outflow)
        hubs = outflow.nlargest(5).to_dict()

        # Information sinks (high inflow)
        sinks = inflow.nlargest(5).to_dict()

        # Strongest directed connections
        strong_connections = []
        for source in value_cols:
            for target in value_cols:
                if source != target:
                    te = te_matrix.loc[source, target]
                    if te > 0.01:  # Threshold
                        strong_connections.append({
                            "source": source,
                            "target": target,
                            "transfer_entropy": float(te)
                        })

        strong_connections.sort(key=lambda x: -x["transfer_entropy"])

        result = {
            "lag": lag,
            "te_matrix": te_matrix.to_dict(),
            "net_transfer": net_transfer,
            "information_hubs": hubs,
            "information_sinks": sinks,
            "strong_connections": strong_connections[:20],
            "mean_te": float(te_matrix.values[te_matrix.values > 0].mean()) if (te_matrix.values > 0).any() else 0,
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def _transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1,
        k: int = 3
    ) -> float:
        """
        Compute transfer entropy from source to target.

        Uses k-nearest neighbors for entropy estimation.
        """
        n = len(source) - lag

        if n < k + 1:
            return 0.0

        try:
            # Create embedding vectors
            # X: target_t, target_{t-1}, source_{t-1}
            # Y: target_t, target_{t-1}
            # Z: target_{t-1}

            target_t = target[lag:].reshape(-1, 1)
            target_past = target[lag-1:-1].reshape(-1, 1)
            source_past = source[lag-1:-1].reshape(-1, 1)

            # Joint spaces
            xyz = np.hstack([target_t, target_past, source_past])
            xy = np.hstack([target_t, target_past])
            yz = np.hstack([target_past, source_past])
            y = target_past

            # Estimate entropies using kNN
            h_xyz = self._knn_entropy(xyz, k)
            h_xy = self._knn_entropy(xy, k)
            h_yz = self._knn_entropy(yz, k)
            h_y = self._knn_entropy(y, k)

            # Transfer entropy = H(X,Y) + H(Y,Z) - H(Y) - H(X,Y,Z)
            te = h_xy + h_yz - h_y - h_xyz

            return max(0, te)  # TE should be non-negative

        except Exception:
            return 0.0

    def _knn_entropy(self, data: np.ndarray, k: int) -> float:
        """Estimate entropy using k-nearest neighbors."""
        n, d = data.shape

        if n <= k:
            return 0.0

        # Find k-nearest neighbor distances
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nn.fit(data)
        distances, _ = nn.kneighbors(data)

        # Use k-th neighbor distance (column k, since column 0 is self)
        rho = distances[:, k]

        # Avoid log(0)
        rho = np.maximum(rho, 1e-10)

        # Entropy estimate
        from scipy.special import digamma
        entropy = d * np.mean(np.log(rho)) + np.log(n - 1) - digamma(k)

        return entropy

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by net information transfer.

        High positive = information source (leading indicator)
        High negative = information sink (lagging indicator)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        net_transfer = result["net_transfer"]

        # Use absolute value for ranking, but keep sign for interpretation
        ranking = pd.DataFrame([
            {"indicator": k, "score": abs(v), "direction": "source" if v > 0 else "sink"}
            for k, v in net_transfer.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)
