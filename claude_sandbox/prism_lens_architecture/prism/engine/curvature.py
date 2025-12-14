# prism/engine/curvature.py
"""
Curvature lens - measures nonlinear bending of trajectory.
"""

import numpy as np
from prism.engine.lens_base import LensBase


class CurvatureLens(LensBase):
    """
    Measures curvature (second derivative) statistics.

    High curvature = sharp turns in trajectory
    Low curvature = linear movement
    """

    name = "curvature"

    def _observe(self, indicator: str, df):
        values = df["value"].dropna().values.astype(float)

        if len(values) < 5:
            return {
                "curvature_mean": 0.0,
                "curvature_std": 0.0,
                "length": int(len(values)),
            }

        # Use gradient for cleaner derivatives
        first = np.gradient(values)
        second = np.gradient(first)

        curvature = np.abs(second)

        return {
            "curvature_mean": float(np.mean(curvature)),
            "curvature_std": float(np.std(curvature)),
            "length": int(len(values)),
        }
