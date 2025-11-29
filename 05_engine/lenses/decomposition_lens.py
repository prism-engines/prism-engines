"""
Decomposition Lens - Time series decomposition

Separates trend, seasonal, and residual components.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import time

from .base_lens import BaseLens


class DecompositionLens(BaseLens):
    """
    Decomposition Lens: Time series decomposition analysis.

    Provides:
    - Trend, seasonal, and residual components
    - Seasonality strength per indicator
    - Trend direction and strength
    """

    name = "decomposition"
    description = "Time series decomposition (trend/seasonal/residual)"
    category = "basic"

    def analyze(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
        model: str = "additive",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Decompose time series into components.

        Args:
            df: Input DataFrame
            period: Seasonal period (default: auto-detect)
            model: 'additive' or 'multiplicative'

        Returns:
            Dictionary with decomposition results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        # Auto-detect period if not specified
        if period is None:
            period = self._detect_period(data, value_cols)

        decomposition_results = {}
        trend_strengths = {}
        seasonal_strengths = {}
        residual_vars = {}

        for col in value_cols:
            series = data[col]

            try:
                from statsmodels.tsa.seasonal import seasonal_decompose

                # Need enough data for decomposition
                if len(series) < 2 * period:
                    continue

                result = seasonal_decompose(
                    series,
                    model=model,
                    period=period,
                    extrapolate_trend='freq'
                )

                # Calculate component strengths
                var_resid = np.var(result.resid.dropna())
                var_original = np.var(series.dropna())
                var_seasonal = np.var(result.seasonal.dropna())
                var_trend = np.var(result.trend.dropna())

                # Strength of trend (R-squared like metric)
                if var_original > 0:
                    trend_strength = var_trend / var_original
                    seasonal_strength = var_seasonal / var_original
                else:
                    trend_strength = 0
                    seasonal_strength = 0

                trend_strengths[col] = float(trend_strength)
                seasonal_strengths[col] = float(seasonal_strength)
                residual_vars[col] = float(var_resid)

                # Trend direction
                trend_vals = result.trend.dropna().values
                if len(trend_vals) > 1:
                    slope, _, r_value, _, _ = stats.linregress(
                        range(len(trend_vals)), trend_vals
                    )
                    trend_direction = "up" if slope > 0 else "down"
                    trend_r2 = r_value ** 2
                else:
                    trend_direction = "flat"
                    trend_r2 = 0

                decomposition_results[col] = {
                    "trend_strength": float(trend_strength),
                    "seasonal_strength": float(seasonal_strength),
                    "trend_direction": trend_direction,
                    "trend_r2": float(trend_r2),
                    "residual_variance": float(var_resid),
                }

            except Exception as e:
                decomposition_results[col] = {"error": str(e)}

        # Identify most seasonal and most trending indicators
        sorted_seasonal = sorted(seasonal_strengths.items(), key=lambda x: -x[1])
        sorted_trend = sorted(trend_strengths.items(), key=lambda x: -x[1])

        result = {
            "period": period,
            "model": model,
            "decomposition_results": decomposition_results,
            "trend_strengths": trend_strengths,
            "seasonal_strengths": seasonal_strengths,
            "most_seasonal": dict(sorted_seasonal[:5]),
            "most_trending": dict(sorted_trend[:5]),
            "mean_trend_strength": float(np.mean(list(trend_strengths.values()))) if trend_strengths else 0,
            "mean_seasonal_strength": float(np.mean(list(seasonal_strengths.values()))) if seasonal_strengths else 0,
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def _detect_period(self, data: pd.DataFrame, value_cols: list) -> int:
        """Auto-detect seasonal period using FFT."""
        try:
            # Use first column with sufficient data
            for col in value_cols:
                series = data[col].dropna()
                if len(series) < 50:
                    continue

                # FFT
                fft_vals = np.fft.fft(series.values)
                freqs = np.fft.fftfreq(len(series))

                # Find dominant frequency (excluding DC component)
                power = np.abs(fft_vals[1:len(fft_vals)//2])
                dominant_idx = np.argmax(power) + 1
                dominant_freq = np.abs(freqs[dominant_idx])

                if dominant_freq > 0:
                    period = int(1 / dominant_freq)
                    # Reasonable bounds
                    period = max(2, min(period, len(series) // 2))
                    return period

        except Exception:
            pass

        # Default periods
        return 12  # Monthly seasonality

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by trend + seasonal strength.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)

        trend = result["trend_strengths"]
        seasonal = result["seasonal_strengths"]

        # Combine trend and seasonal strength
        combined = {}
        all_cols = set(trend.keys()) | set(seasonal.keys())
        for col in all_cols:
            combined[col] = trend.get(col, 0) + seasonal.get(col, 0)

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in combined.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)
