# prism/engine/geometry.py
"""
Geometry lens - basic statistical geometry of a time series.
"""

import numpy as np
from prism.engine.lens_base import LensBase


class GeometryLens(LensBase):
    """
    Basic geometric properties: mean, std, slope.

    The foundation lens that all other lenses can reference.
    """

    name = "geometry"

    def _observe(self, indicator: str, df):
        values = df["value"].dropna().values.astype(float)

        if len(values) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "slope_mean": 0.0,
                "length": 0,
            }

        diffs = np.diff(values) if len(values) > 1 else np.array([0.0])

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "slope_mean": float(np.mean(diffs)),
            "length": int(len(values)),
        }
