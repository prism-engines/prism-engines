"""
Hurst Engine - Long-term memory and persistence analysis.
"""

import numpy as np
import pandas as pd
from .base import BaseEngine, EngineResult


class HurstEngine(BaseEngine):
    """
    Computes Hurst exponent for each column using R/S analysis.

    Interpretation:
        H < 0.5: Anti-persistent (mean-reverting)
        H = 0.5: Random walk
        H > 0.5: Persistent (trending)

    Metrics:
        - hurst_exponents: H value per column
        - mean_hurst: Average H across all columns
        - persistence_classification: Categorization of each column
    """

    name = "hurst"
    description = "Long-term memory analysis via Hurst exponent"

    def run(self, df: pd.DataFrame) -> EngineResult:
        error = self.validate(df)
        if error:
            return EngineResult(self.name, error=error)

        try:
            hurst_values = {}
            classifications = {}

            for col in df.columns:
                series = df[col].dropna().values
                if len(series) < 20:
                    hurst_values[col] = None
                    classifications[col] = "insufficient_data"
                    continue

                h = self._compute_hurst(series)
                hurst_values[col] = h

                if h < 0.4:
                    classifications[col] = "anti_persistent"
                elif h < 0.6:
                    classifications[col] = "random_walk"
                else:
                    classifications[col] = "persistent"

            valid_h = [v for v in hurst_values.values() if v is not None]
            mean_h = float(np.mean(valid_h)) if valid_h else None

            return EngineResult(
                engine_name=self.name,
                metrics={
                    "hurst_exponents": hurst_values,
                    "mean_hurst": mean_h,
                    "persistence_classification": classifications,
                },
                metadata={"n_analyzed": len(valid_h)},
            )

        except Exception as e:
            return EngineResult(self.name, error=str(e))

    def _compute_hurst(self, series: np.ndarray) -> float:
        """
        Compute Hurst exponent using R/S analysis.
        """
        n = len(series)
        max_k = min(n // 2, 100)

        if max_k < 8:
            return 0.5  # Default for short series

        rs_values = []
        sizes = []

        for size in range(8, max_k + 1, max(1, max_k // 20)):
            n_segments = n // size
            if n_segments < 1:
                continue

            rs_list = []
            for i in range(n_segments):
                segment = series[i * size : (i + 1) * size]
                mean = np.mean(segment)
                devs = segment - mean
                cumsum = np.cumsum(devs)
                r = np.max(cumsum) - np.min(cumsum)
                s = np.std(segment, ddof=1)
                if s > 1e-10:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                sizes.append(size)

        if len(sizes) < 3:
            return 0.5

        # Linear regression in log-log space
        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)

        slope, _ = np.polyfit(log_sizes, log_rs, 1)

        return float(np.clip(slope, 0, 1))
