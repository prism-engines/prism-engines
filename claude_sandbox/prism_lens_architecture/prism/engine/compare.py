# prism/engine/compare.py
"""
Compare lens - composite metrics for cross-indicator comparison.
"""

import numpy as np
from prism.engine.lens_base import LensBase


class CompareLens(LensBase):
    """
    Composite metrics useful for comparing indicators.

    Combines basic stats with derived ratios.
    """

    name = "compare"

    def _observe(self, indicator: str, df):
        values = df["value"].dropna().values.astype(float)

        if len(values) < 3:
            return {
                "mean": 0.0,
                "std": 0.0,
                "gap": 0.0,
                "volatility_ratio": 0.0,
                "length": int(len(values)),
            }

        diffs = np.diff(values)
        curvature = np.diff(diffs)

        std_val = np.std(values)
        std_diff = np.std(diffs)

        return {
            "mean": float(np.mean(values)),
            "std": float(std_val),
            "gap": float(np.std(curvature)),
            "volatility_ratio": float(std_diff / (std_val + 1e-9)),
            "length": int(len(values)),
        }
