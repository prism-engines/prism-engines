"""
PCA Engine - Principal Component Analysis.
"""

import numpy as np
import pandas as pd
from .base import BaseEngine, EngineResult


class PCAEngine(BaseEngine):
    """
    Computes principal components and effective dimensionality.

    Metrics:
        - explained_variance_ratio: Variance explained by each PC
        - effective_dimension: Number of PCs to explain 90% variance
        - pc1_loadings: Column loadings on first principal component
        - global_forcing_metric: PC1 variance / total (how dominant is PC1)
    """

    name = "pca"
    description = "Principal component analysis"

    def run(self, df: pd.DataFrame) -> EngineResult:
        error = self.validate(df)
        if error:
            return EngineResult(self.name, error=error)

        try:
            # Standardize
            X = df.values
            X_centered = X - X.mean(axis=0)
            X_std = X_centered / (X_centered.std(axis=0) + 1e-10)

            # SVD for PCA
            U, S, Vt = np.linalg.svd(X_std, full_matrices=False)

            # Explained variance
            variance = S ** 2 / (len(X) - 1)
            total_var = variance.sum()
            explained_ratio = variance / total_var if total_var > 0 else variance

            # Effective dimension (PCs to explain 90%)
            cumsum = np.cumsum(explained_ratio)
            effective_dim = int(np.searchsorted(cumsum, 0.9) + 1)

            # PC1 loadings
            pc1_loadings = dict(zip(df.columns, Vt[0].tolist()))

            # Global forcing metric
            gfm = float(explained_ratio[0]) if len(explained_ratio) > 0 else 0.0

            return EngineResult(
                engine_name=self.name,
                metrics={
                    "explained_variance_ratio": explained_ratio[:5].tolist(),
                    "effective_dimension": effective_dim,
                    "pc1_loadings": pc1_loadings,
                    "global_forcing_metric": gfm,
                    "cumulative_variance": cumsum[:5].tolist(),
                },
                metadata={
                    "n_components": len(variance),
                    "total_variance": float(total_var),
                },
            )

        except Exception as e:
            return EngineResult(self.name, error=str(e))
