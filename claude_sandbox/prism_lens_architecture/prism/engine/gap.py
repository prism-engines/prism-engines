# prism/engine/gap.py
"""
Gap lens - measures curvature volatility (second derivative behavior).
"""

import numpy as np
from prism.engine.lens_base import LensBase


class GapLens(LensBase):
    """
    Measures gap/curvature volatility.

    High gap_score = erratic curvature changes
    Low gap_score = smooth trajectory
    """

    name = "gap"

    def _observe(self, indicator: str, df):
        values = df["value"].dropna().values.astype(float)

        if len(values) < 3:
            return {
                "gap_score": 0.0,
                "length": int(len(values)),
            }

        diffs = np.diff(values)
        curvature = np.diff(diffs)

        gap_score = float(np.std(curvature))

        return {
            "gap_score": gap_score,
            "length": int(len(values)),
        }
