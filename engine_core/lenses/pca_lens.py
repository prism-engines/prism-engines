"""
PCA Lens - Principal Component Analysis

Identifies the main drivers of variance in the indicator space.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import time

from .base_lens import BaseLens


class PCALens(BaseLens):
    """
    PCA Lens: Principal Component Analysis of indicators.

    Provides:
    - Principal components and explained variance
    - Indicator loadings on each component
    - Dimensionality reduction insights
    """

    name = "pca"
    description = "Principal Component Analysis for variance decomposition"
    category = "basic"

    def analyze(
        self,
        df: pd.DataFrame,
        n_components: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run PCA analysis.

        Args:
            df: Input DataFrame
            n_components: Number of components (default: all)

        Returns:
            Dictionary with PCA results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        data = self.normalize_data(data)
        value_cols = self.get_value_columns(data)

        values = data[value_cols].values

        # Fit PCA
        n_components = n_components or min(len(value_cols), len(values))
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(values)

        # Loadings matrix
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=value_cols
        )

        # Find top contributors to each PC
        top_contributors = {}
        for pc in loadings.columns:
            sorted_loadings = loadings[pc].abs().sort_values(ascending=False)
            top_contributors[pc] = sorted_loadings.head(5).index.tolist()

        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # How many components to explain 80% and 95%?
        n_for_80 = int(np.searchsorted(cumulative_var, 0.80) + 1)
        n_for_95 = int(np.searchsorted(cumulative_var, 0.95) + 1)

        result = {
            "n_components": n_components,
            "explained_variance_ratio": explained_var.tolist(),
            "cumulative_variance": cumulative_var.tolist(),
            "n_components_80pct": n_for_80,
            "n_components_95pct": n_for_95,
            "loadings": loadings.to_dict(),
            "top_contributors": top_contributors,
            "pc1_variance": float(explained_var[0]),
            "pc2_variance": float(explained_var[1]) if len(explained_var) > 1 else 0,
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by their contribution to principal components.

        Uses sum of squared loadings across top components as importance.

        Args:
            df: Input DataFrame
            n_top_pcs: Number of PCs to consider (default: 3)

        Returns:
            DataFrame with indicator rankings
        """
        n_top_pcs = kwargs.get("n_top_pcs", 3)
        result = self.analyze(df, **kwargs)
        loadings = pd.DataFrame(result["loadings"])

        # Use top N PCs
        top_pcs = loadings.columns[:n_top_pcs]

        # Weight by explained variance
        var_ratios = result["explained_variance_ratio"][:n_top_pcs]

        # Compute weighted importance
        importance = {}
        for indicator in loadings.index:
            weighted_sum = 0
            for i, pc in enumerate(top_pcs):
                weighted_sum += (loadings.loc[indicator, pc] ** 2) * var_ratios[i]
            importance[indicator] = weighted_sum

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in importance.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)

    def get_loadings_heatmap_data(self, df: pd.DataFrame, n_pcs: int = 5) -> Dict:
        """
        Get data for loadings heatmap visualization.

        Args:
            df: Input DataFrame
            n_pcs: Number of PCs to include

        Returns:
            Dictionary with heatmap data
        """
        result = self.analyze(df, n_components=n_pcs)
        loadings = pd.DataFrame(result["loadings"])

        return {
            "indicators": loadings.index.tolist(),
            "components": loadings.columns.tolist(),
            "values": loadings.values.tolist(),
        }
