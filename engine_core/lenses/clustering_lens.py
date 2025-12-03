"""
Clustering Lens - Hierarchical clustering of indicators

Groups indicators by similarity in behavior.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import time

from .base_lens import BaseLens


class ClusteringLens(BaseLens):
    """
    Clustering Lens: Hierarchical clustering of indicators.

    Provides:
    - Indicator clusters based on correlation/behavior
    - Cluster representatives (centroids)
    - Dendrogram structure
    """

    name = "clustering"
    description = "Hierarchical clustering of indicators by behavior"
    category = "basic"

    def analyze(
        self,
        df: pd.DataFrame,
        n_clusters: Optional[int] = None,
        method: str = "ward",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform hierarchical clustering.

        Args:
            df: Input DataFrame
            n_clusters: Number of clusters (default: auto-determined)
            method: Linkage method ('ward', 'complete', 'average', 'single')

        Returns:
            Dictionary with clustering results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        data = self.normalize_data(data)
        value_cols = self.get_value_columns(data)

        # Transpose: each indicator is a sample
        indicator_matrix = data[value_cols].T.values

        # Compute distance matrix
        metric = kwargs.get("metric", "correlation")
        if metric == "correlation":
            # Use 1 - correlation as distance
            corr_matrix = np.corrcoef(indicator_matrix)
            dist_matrix = 1 - corr_matrix
            np.fill_diagonal(dist_matrix, 0)
            condensed_dist = squareform(dist_matrix)
        else:
            condensed_dist = pdist(indicator_matrix, metric=metric)

        # Hierarchical clustering
        Z = linkage(condensed_dist, method=method)

        # Determine n_clusters if not specified
        if n_clusters is None:
            # Use elbow method or fixed fraction
            n_clusters = max(2, len(value_cols) // 5)

        # Get cluster assignments
        labels = fcluster(Z, n_clusters, criterion='maxclust')

        # Build cluster membership
        clusters = {}
        for i, col in enumerate(value_cols):
            cluster_id = int(labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(col)

        # Compute cluster centroids (mean of member series)
        centroids = {}
        for cluster_id, members in clusters.items():
            centroid = data[members].mean(axis=1).values
            centroids[cluster_id] = centroid.tolist()

        # Find representative indicator per cluster (closest to centroid)
        representatives = {}
        for cluster_id, members in clusters.items():
            if len(members) == 1:
                representatives[cluster_id] = members[0]
            else:
                centroid = np.array(centroids[cluster_id])
                min_dist = float('inf')
                rep = members[0]
                for member in members:
                    dist = np.linalg.norm(data[member].values - centroid)
                    if dist < min_dist:
                        min_dist = dist
                        rep = member
                representatives[cluster_id] = rep

        # Cluster cohesion (within-cluster correlation)
        cluster_cohesion = {}
        for cluster_id, members in clusters.items():
            if len(members) > 1:
                corrs = data[members].corr().values
                mask = np.triu(np.ones_like(corrs, dtype=bool), k=1)
                cohesion = np.mean(corrs[mask])
            else:
                cohesion = 1.0
            cluster_cohesion[cluster_id] = float(cohesion)

        result = {
            "n_clusters": n_clusters,
            "method": method,
            "clusters": clusters,
            "cluster_sizes": {k: len(v) for k, v in clusters.items()},
            "representatives": representatives,
            "cluster_cohesion": cluster_cohesion,
            "linkage_matrix": Z.tolist(),
            "indicator_labels": dict(zip(value_cols, labels.tolist())),
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by their centrality within clusters.

        Representatives get highest scores.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)

        clusters = result["clusters"]
        representatives = result["representatives"]
        cohesion = result["cluster_cohesion"]

        scores = {}

        for cluster_id, members in clusters.items():
            cluster_cohesion = cohesion.get(cluster_id, 0.5)
            cluster_size = len(members)

            for member in members:
                # Score based on cluster size and cohesion
                base_score = cluster_size * cluster_cohesion

                # Representatives get bonus
                if representatives.get(cluster_id) == member:
                    base_score *= 1.5

                scores[member] = base_score

        # Normalize
        max_score = max(scores.values()) if scores else 1
        scores = {k: v / max_score for k, v in scores.items()}

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in scores.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)

    def get_dendrogram_data(self, df: pd.DataFrame, **kwargs) -> Dict:
        """
        Get data for dendrogram visualization.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with dendrogram data
        """
        result = self.analyze(df, **kwargs)

        return {
            "linkage_matrix": result["linkage_matrix"],
            "labels": list(result["indicator_labels"].keys()),
        }
