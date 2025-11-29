"""
Magnitude Lens - Simple vector magnitude analysis

Measures the overall "energy" or movement in the indicator space.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import time

from .base_lens import BaseLens


class MagnitudeLens(BaseLens):
    """
    Magnitude Lens: Analyze vector magnitudes in indicator space.

    Provides:
    - Overall market state magnitude (Euclidean norm)
    - Per-indicator contribution to magnitude
    - Magnitude trends and anomalies
    """

    name = "magnitude"
    description = "Vector magnitude analysis in indicator space"
    category = "basic"

    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyze vector magnitudes.

        Args:
            df: Input DataFrame
            **kwargs:
                normalize: Whether to normalize data (default True)
                window: Rolling window for trends (default 20)

        Returns:
            Dictionary with magnitude analysis
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        normalize = kwargs.get("normalize", True)
        window = kwargs.get("window", 20)

        if normalize:
            data = self.normalize_data(data)

        # Extract value matrix
        values = data[value_cols].values

        # Compute magnitudes for each time point
        magnitudes = np.linalg.norm(values, axis=1)

        # Per-indicator contribution (squared contribution to total)
        contributions = {}
        total_energy = np.sum(values ** 2, axis=0)
        for i, col in enumerate(value_cols):
            contributions[col] = total_energy[i] / np.sum(total_energy)

        # Rolling statistics
        mag_series = pd.Series(magnitudes, index=data["date"])
        rolling_mean = mag_series.rolling(window).mean()
        rolling_std = mag_series.rolling(window).std()

        # Detect magnitude spikes (>2 std from rolling mean)
        z_score = (mag_series - rolling_mean) / rolling_std
        spikes = data.loc[z_score.abs() > 2, "date"].tolist()

        result = {
            "current_magnitude": float(magnitudes[-1]) if len(magnitudes) > 0 else 0,
            "mean_magnitude": float(np.mean(magnitudes)),
            "std_magnitude": float(np.std(magnitudes)),
            "max_magnitude": float(np.max(magnitudes)),
            "min_magnitude": float(np.min(magnitudes)),
            "indicator_contributions": contributions,
            "magnitude_trend": float(rolling_mean.iloc[-1]) if len(rolling_mean) > 0 else 0,
            "n_spikes": len(spikes),
            "spike_dates": [str(d) for d in spikes[-10:]],  # Last 10 spikes
            "time_series": {
                "dates": [str(d) for d in data["date"].tolist()],
                "magnitudes": magnitudes.tolist(),
            }
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by their contribution to overall magnitude.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        contributions = result["indicator_contributions"]

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in contributions.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)
