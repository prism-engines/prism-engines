# prism/engine/changepoint.py
"""
Changepoint lens - detects structural breaks in time series.
"""

import numpy as np
from prism.engine.lens_base import LensBase


class ChangepointLens(LensBase):
    """
    Detects changepoints (structural breaks).

    High change_score = large jumps detected
    High num_changes = many structural breaks
    """

    name = "changepoint"

    def _observe(self, indicator: str, df):
        values = df["value"].dropna().values.astype(float)

        if len(values) < 5:
            return {
                "change_score": 0.0,
                "num_changes": 0,
                "length": int(len(values)),
            }

        diffs = np.abs(np.diff(values))
        threshold = np.mean(diffs) + 2 * np.std(diffs)
        change_points = np.where(diffs > threshold)[0]

        return {
            "change_score": float(diffs.max()),
            "num_changes": int(len(change_points)),
            "length": int(len(values)),
        }
