"""
PRISM Clustering Engine

Groups indicators by behavioral similarity.

Measures:
- Cluster assignments
- Cluster centroids
- Silhouette scores
- Optimal cluster count

Phase: Structure
Normalization: Z-score
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from .base import BaseEngine


logger = logging.getLogger(__name__)


class ClusteringEngine(BaseEngine):
    """
    Clustering engine for behavioral grouping.
    
    Groups indicators based on correlation structure or return patterns.
    
    Outputs:
        - structure.clusters: Indicator cluster assignments
        - structure.centroids: Cluster centroids
    """
    
    name = "clustering"
    phase = "structure"
    default_normalization = "zscore"
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_clusters: Optional[int] = None,
        method: str = "kmeans",
        max_clusters: int = 10,
        use_correlation: bool = True,
        **params
    ) -> Dict[str, Any]:
        """
        Run clustering analysis.
        
        Args:
            df: Normalized indicator data
            run_id: Unique run identifier
            n_clusters: Number of clusters (None = auto-detect)
            method: 'kmeans' or 'hierarchical'
            max_clusters: Max clusters for auto-detection
            use_correlation: If True, cluster on correlation matrix
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        n_indicators = len(indicators)
        
        if n_indicators < 3:
            raise ValueError(f"Need at least 3 indicators for clustering, got {n_indicators}")
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        # Prepare feature matrix
        if use_correlation:
            # Use correlation structure as features
            corr_matrix = df_clean.corr()
            X = corr_matrix.values
        else:
            # Use transposed data (indicators as samples)
            X = df_clean.T.values
        
        # Auto-detect optimal clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X, max_clusters, method)
        
        n_clusters = min(n_clusters, n_indicators - 1)
        
        # Fit clustering
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            centroids = model.cluster_centers_
        else:  # hierarchical
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
            # Compute centroids manually
            centroids = np.array([
                X[labels == i].mean(axis=0) for i in range(n_clusters)
            ])
        
        # Compute quality metrics
        if n_clusters > 1 and n_clusters < n_indicators:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
        else:
            silhouette = 0
            calinski = 0
        
        # Store results
        self._store_clusters(
            indicators, labels, window_start, window_end, run_id
        )
        self._store_centroids(
            centroids, window_start, window_end, run_id
        )
        
        # Cluster composition
        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        
        metrics = {
            "n_indicators": n_indicators,
            "n_clusters": int(n_clusters),
            "method": method,
            "silhouette_score": float(silhouette),
            "calinski_harabasz": float(calinski),
            "cluster_sizes": cluster_sizes,
            "use_correlation": use_correlation,
        }
        
        logger.info(
            f"Clustering complete: {n_clusters} clusters, "
            f"silhouette={silhouette:.3f}"
        )
        
        return metrics
    
    def _find_optimal_clusters(
        self,
        X: np.ndarray,
        max_clusters: int,
        method: str
    ) -> int:
        """Find optimal cluster count using silhouette score."""
        n_samples = X.shape[0]
        max_k = min(max_clusters, n_samples - 1)
        
        if max_k < 2:
            return 2
        
        scores = []
        for k in range(2, max_k + 1):
            if method == "kmeans":
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)
            
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append((k, score))
        
        # Return k with highest silhouette
        best_k = max(scores, key=lambda x: x[1])[0]
        return best_k
    
    def _store_clusters(
        self,
        indicators: list,
        labels: np.ndarray,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store cluster assignments to structure.clusters."""
        records = []
        for indicator, label in zip(indicators, labels):
            records.append({
                "cluster_id": f"cluster_{label}",
                "window_start": window_start,
                "window_end": window_end,
                "indicator_id": indicator,
                "membership": 1.0,  # Hard clustering
                "run_id": run_id,
            })
        
        df = pd.DataFrame(records)
        self.store_results("clusters", df, run_id)
    
    def _store_centroids(
        self,
        centroids: np.ndarray,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store cluster centroids to structure.centroids."""
        records = []
        for i, centroid in enumerate(centroids):
            for j, value in enumerate(centroid):
                records.append({
                    "cluster_id": f"cluster_{i}",
                    "window_start": window_start,
                    "window_end": window_end,
                    "dimension": f"dim_{j}",
                    "value": float(value),
                    "run_id": run_id,
                })
        
        df = pd.DataFrame(records)
        self.store_results("centroids", df, run_id)
