"""
Correlation Engine - Pairwise correlation analysis.
"""

import numpy as np
import pandas as pd
from .base import BaseEngine, EngineResult


class CorrelationEngine(BaseEngine):
    """
    Computes pairwise correlations between all columns.

    Metrics:
        - correlation_matrix: Full NxN correlation matrix
        - mean_correlation: Average absolute correlation
        - max_correlation: Strongest correlation pair
        - min_correlation: Weakest correlation pair
    """

    name = "correlation"
    description = "Pairwise correlation analysis"

    def run(self, df: pd.DataFrame) -> EngineResult:
        error = self.validate(df)
        if error:
            return EngineResult(self.name, error=error)

        try:
            # Compute correlation matrix
            corr_matrix = df.corr().values
            n = len(df.columns)

            # Extract upper triangle (excluding diagonal)
            upper_idx = np.triu_indices(n, k=1)
            upper_vals = corr_matrix[upper_idx]

            # Find strongest and weakest pairs
            abs_vals = np.abs(upper_vals)
            max_idx = np.argmax(abs_vals)
            min_idx = np.argmin(abs_vals)

            cols = list(df.columns)
            max_pair = (cols[upper_idx[0][max_idx]], cols[upper_idx[1][max_idx]])
            min_pair = (cols[upper_idx[0][min_idx]], cols[upper_idx[1][min_idx]])

            return EngineResult(
                engine_name=self.name,
                metrics={
                    "correlation_matrix": corr_matrix.tolist(),
                    "mean_abs_correlation": float(np.mean(abs_vals)),
                    "max_correlation": {
                        "pair": max_pair,
                        "value": float(upper_vals[max_idx]),
                    },
                    "min_correlation": {
                        "pair": min_pair,
                        "value": float(upper_vals[min_idx]),
                    },
                },
                metadata={"n_pairs": len(upper_vals), "columns": cols},
            )

        except Exception as e:
            return EngineResult(self.name, error=str(e))
