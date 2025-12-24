"""
PRISM DTW (Dynamic Time Warping) Engine

Measures shape similarity between time series.

Measures:
- DTW distance matrix
- Optimal warping paths
- Cluster-based similarity groupings

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


def _dtw_distance(x: np.ndarray, y: np.ndarray, window: Optional[int] = None) -> float:
    """
    Compute DTW distance between two sequences.

    Uses dynamic programming with optional Sakoe-Chiba band constraint.
    """
    n, m = len(x), len(y)

    # Initialize cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Sakoe-Chiba band
    if window is None:
        window = max(n, m)

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],      # deletion
                dtw[i - 1, j - 1]   # match
            )

    return np.sqrt(dtw[n, m])


class DTWEngine(BaseEngine):
    """
    Dynamic Time Warping engine.

    Measures shape similarity between time series, allowing for
    temporal warping/alignment.

    Outputs:
        - derived.dtw_distances: Pairwise DTW distance matrix
        - derived.dtw_clusters: Shape-based cluster assignments
    """

    name = "dtw"
    phase = "derived"
    default_normalization = "zscore"

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        window_constraint: Optional[int] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run DTW analysis.

        Args:
            df: Normalized indicator data
            run_id: Unique run identifier
            window_constraint: Sakoe-Chiba band width (None = no constraint)

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = df_clean.columns.tolist()
        n_indicators = len(indicators)

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Compute pairwise DTW distances
        distances = np.zeros((n_indicators, n_indicators))

        for i in range(n_indicators):
            for j in range(i + 1, n_indicators):
                x = df_clean.iloc[:, i].values
                y = df_clean.iloc[:, j].values

                dist = _dtw_distance(x, y, window=window_constraint)
                distances[i, j] = dist
                distances[j, i] = dist

        # Convert to similarity (for comparison with correlation)
        max_dist = distances.max() if distances.max() > 0 else 1.0
        similarity = 1 - (distances / max_dist)

        # Store distance matrix
        records = []
        for i, ind_i in enumerate(indicators):
            for j, ind_j in enumerate(indicators):
                if i < j:
                    records.append({
                        "indicator_1": ind_i,
                        "indicator_2": ind_j,
                        "window_start": window_start,
                        "window_end": window_end,
                        "dtw_distance": float(distances[i, j]),
                        "dtw_similarity": float(similarity[i, j]),
                        "run_id": run_id,
                    })

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("dtw_distances", df_results, run_id)

        # Summary metrics
        n_pairs = n_indicators * (n_indicators - 1) // 2
        condensed = squareform(distances)

        metrics = {
            "n_indicators": n_indicators,
            "n_pairs": n_pairs,
            "n_samples": len(df_clean),
            "avg_dtw_distance": float(np.mean(condensed)) if len(condensed) > 0 else 0.0,
            "max_dtw_distance": float(np.max(condensed)) if len(condensed) > 0 else 0.0,
            "min_dtw_distance": float(np.min(condensed)) if len(condensed) > 0 else 0.0,
            "avg_similarity": float(np.mean(similarity[np.triu_indices(n_indicators, k=1)])) if n_pairs > 0 else 0.0,
            "window_constraint": window_constraint,
        }

        logger.info(
            f"DTW complete: {n_indicators} indicators, "
            f"avg_dist={metrics['avg_dtw_distance']:.4f}"
        )

        return metrics
