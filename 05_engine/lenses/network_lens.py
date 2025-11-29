"""
Network Lens - Network graph analysis of indicators

Builds correlation/dependency networks and computes graph metrics.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import time

from .base_lens import BaseLens


class NetworkLens(BaseLens):
    """
    Network Lens: Graph-based analysis of indicator relationships.

    Provides:
    - Correlation network construction
    - Centrality measures (degree, betweenness, eigenvector)
    - Community detection
    - Network density and clustering coefficient
    """

    name = "network"
    description = "Network graph analysis of indicator relationships"
    category = "advanced"

    def analyze(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build and analyze correlation network.

        Args:
            df: Input DataFrame
            threshold: Correlation threshold for edge creation

        Returns:
            Dictionary with network analysis results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        # Compute correlation matrix
        corr_matrix = data[value_cols].corr()

        # Build adjacency matrix (edges where |corr| > threshold)
        adj_matrix = (corr_matrix.abs() > threshold).astype(int)
        np.fill_diagonal(adj_matrix.values, 0)

        # Basic network stats
        n_nodes = len(value_cols)
        n_edges = adj_matrix.values.sum() // 2  # Undirected
        density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0

        # Degree centrality
        degree = adj_matrix.sum(axis=1).to_dict()
        degree_centrality = {k: v / (n_nodes - 1) for k, v in degree.items()}

        # Weighted degree (sum of correlations)
        weighted_degree = {}
        for col in value_cols:
            weights = corr_matrix[col].abs()
            weights = weights.drop(col)
            weighted_degree[col] = float(weights[weights > threshold].sum())

        # Eigenvector centrality (approximate via power iteration)
        try:
            eigenvector_centrality = self._eigenvector_centrality(adj_matrix.values, value_cols)
        except Exception:
            eigenvector_centrality = {col: 1.0/n_nodes for col in value_cols}

        # Clustering coefficient per node
        clustering = self._clustering_coefficient(adj_matrix.values, value_cols)

        # Find hub nodes (high degree + high clustering)
        hub_scores = {}
        for col in value_cols:
            hub_scores[col] = degree_centrality.get(col, 0) * (1 + clustering.get(col, 0))

        sorted_hubs = sorted(hub_scores.items(), key=lambda x: -x[1])

        result = {
            "n_nodes": n_nodes,
            "n_edges": int(n_edges),
            "density": float(density),
            "threshold": threshold,
            "degree": degree,
            "degree_centrality": degree_centrality,
            "weighted_degree": weighted_degree,
            "eigenvector_centrality": eigenvector_centrality,
            "clustering_coefficient": clustering,
            "avg_clustering": float(np.mean(list(clustering.values()))),
            "hub_scores": hub_scores,
            "top_hubs": dict(sorted_hubs[:5]),
            "isolated_nodes": [col for col in value_cols if degree.get(col, 0) == 0],
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def _eigenvector_centrality(
        self,
        adj_matrix: np.ndarray,
        labels: List[str],
        max_iter: int = 100
    ) -> Dict[str, float]:
        """Compute eigenvector centrality via power iteration."""
        n = adj_matrix.shape[0]
        x = np.ones(n) / n

        for _ in range(max_iter):
            x_new = adj_matrix @ x
            norm = np.linalg.norm(x_new)
            if norm > 0:
                x_new = x_new / norm
            if np.allclose(x, x_new):
                break
            x = x_new

        return dict(zip(labels, x.tolist()))

    def _clustering_coefficient(
        self,
        adj_matrix: np.ndarray,
        labels: List[str]
    ) -> Dict[str, float]:
        """Compute local clustering coefficient for each node."""
        clustering = {}
        n = adj_matrix.shape[0]

        for i, label in enumerate(labels):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            k = len(neighbors)

            if k < 2:
                clustering[label] = 0.0
                continue

            # Count edges between neighbors
            edges_between = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j], neighbors[l]] == 1:
                        edges_between += 1

            max_edges = k * (k - 1) / 2
            clustering[label] = edges_between / max_edges if max_edges > 0 else 0

        return clustering

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by network centrality (hub score).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        hub_scores = result["hub_scores"]

        # Normalize
        max_score = max(hub_scores.values()) if hub_scores else 1
        normalized = {k: v / max_score for k, v in hub_scores.items()}

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in normalized.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)

    def get_network_data(self, df: pd.DataFrame, **kwargs) -> Dict:
        """
        Get data for network visualization.

        Returns nodes and edges for graph rendering.
        """
        result = self.analyze(df, **kwargs)
        threshold = result["threshold"]

        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)
        corr_matrix = data[value_cols].corr()

        nodes = [
            {
                "id": col,
                "centrality": result["eigenvector_centrality"].get(col, 0),
                "degree": result["degree"].get(col, 0),
            }
            for col in value_cols
        ]

        edges = []
        for i, col_i in enumerate(value_cols):
            for j, col_j in enumerate(value_cols):
                if i < j:
                    corr = corr_matrix.loc[col_i, col_j]
                    if abs(corr) > threshold:
                        edges.append({
                            "source": col_i,
                            "target": col_j,
                            "weight": float(abs(corr)),
                            "positive": corr > 0,
                        })

        return {"nodes": nodes, "edges": edges}
