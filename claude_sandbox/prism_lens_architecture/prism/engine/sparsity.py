# prism/engine/sparsity.py
"""
Sparsity lens - measures jump behavior relative to volatility.
"""

import numpy as np
from prism.engine.lens_base import LensBase


class SparsityLens(LensBase):
    """
    Measures sparsity (average jump magnitude scaled by volatility).

    High sparsity = large movements relative to baseline
    Low sparsity = smooth, low-jump behavior
    """

    name = "sparsity"

    def _observe(self, indicator: str, df):
        values = df["value"].dropna().values.astype(float)

        if len(values) < 3:
            return {
                "sparsity_score": 0.0,
                "mean_jump": 0.0,
                "length": int(len(values)),
            }

        # First differences
        diffs = np.diff(values)

        # Absolute change magnitudes
        mags = np.abs(diffs)

        # Sparsity = mean jump scaled by volatility
        mean_jump = np.mean(mags)
        std = np.std(values)

        sparsity = mean_jump / std if std > 0 else 0.0

        return {
            "sparsity_score": float(sparsity),
            "mean_jump": float(mean_jump),
            "length": int(len(values)),
        }
