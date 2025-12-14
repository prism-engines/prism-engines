# prism/engine/entropy.py
"""
Entropy lens - measures distributional complexity of a time series.
"""

import numpy as np
from prism.engine.lens_base import LensBase


class EntropyLens(LensBase):
    """
    Measures entropy (distributional complexity) of values.

    High entropy = values spread across range
    Low entropy = values concentrated
    """

    name = "entropy"

    def _observe(self, indicator: str, df):
        values = df["value"].dropna().values.astype(float)

        if len(values) < 2:
            return {
                "entropy": 0.0,
                "length": len(values),
            }

        # Histogram-based entropy
        hist, _ = np.histogram(values, bins=20, density=True)
        hist = hist[hist > 0]

        if len(hist) == 0:
            entropy = 0.0
        else:
            entropy = -np.sum(hist * np.log(hist))

        return {
            "entropy": float(entropy),
            "length": int(len(values)),
        }
