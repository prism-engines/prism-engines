"""
Wavelet Lens - Wavelet transform analysis

Multi-scale time-frequency analysis of indicators.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import time

from .base_lens import BaseLens


class WaveletLens(BaseLens):
    """
    Wavelet Lens: Multi-scale wavelet analysis.

    Provides:
    - Wavelet coefficients at multiple scales
    - Energy distribution across scales
    - Scale-specific correlations
    """

    name = "wavelet"
    description = "Wavelet transform for multi-scale analysis"
    category = "advanced"

    def analyze(
        self,
        df: pd.DataFrame,
        wavelet: str = "db4",
        max_level: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform wavelet decomposition.

        Args:
            df: Input DataFrame
            wavelet: Wavelet type ('db4', 'haar', 'sym4', etc.)
            max_level: Maximum decomposition level

        Returns:
            Dictionary with wavelet analysis results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        try:
            import pywt
        except ImportError:
            return {"error": "pywt not installed. Run: pip install PyWavelets"}

        # Determine max level
        if max_level is None:
            max_level = pywt.dwt_max_level(len(data), wavelet)
            max_level = min(max_level, 8)  # Cap at 8 levels

        energy_by_scale = {f"level_{i}": {} for i in range(1, max_level + 1)}
        energy_by_scale["approx"] = {}

        total_energy = {}
        dominant_scales = {}
        dominant_periods = {}
        significant_periods = set()  # All significant periods across columns

        for col in value_cols:
            series = data[col].values

            # Wavelet decomposition
            coeffs = pywt.wavedec(series, wavelet, level=max_level)

            # Compute energy at each scale
            energies = []
            for i, c in enumerate(coeffs):
                energy = np.sum(c ** 2)
                if i == 0:
                    energy_by_scale["approx"][col] = float(energy)
                else:
                    energy_by_scale[f"level_{i}"][col] = float(energy)
                energies.append(energy)

            total_energy[col] = float(sum(energies))

            # Find dominant scale
            if energies:
                dominant_idx = np.argmax(energies)
                if dominant_idx == 0:
                    dominant_scales[col] = "approx (long-term trend)"
                    dominant_periods[col] = len(series)
                else:
                    dominant_scales[col] = f"level_{dominant_idx} (scale ~{2**dominant_idx})"
                    dominant_periods[col] = 2 ** dominant_idx
                # Find ALL significant periods (>5% of max energy, excluding approx)
                max_energy = max(energies)
                for i, e in enumerate(energies):
                    if i > 0 and e > max_energy * 0.01:  # Skip approx
                        significant_periods.add(2 ** i)

        # Scale correlations (how correlated are indicators at each scale)
        scale_correlations = {}
        for level_name in energy_by_scale:
            level_energies = energy_by_scale[level_name]
            if level_energies:
                values = list(level_energies.values())
                if len(values) > 1:
                    scale_correlations[level_name] = float(np.std(values) / (np.mean(values) + 1e-10))

        result = {
            "wavelet": wavelet,
            "max_level": max_level,
            "energy_by_scale": energy_by_scale,
            "total_energy": total_energy,
            "dominant_scales": dominant_scales,
            "dominant_periods": dominant_periods,
            "significant_periods": sorted(significant_periods),
            "scale_variability": scale_correlations,
            "n_indicators": len(value_cols),
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by total wavelet energy.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)

        if "error" in result:
            return pd.DataFrame(columns=["indicator", "score", "rank"])

        total_energy = result["total_energy"]

        # Normalize
        max_energy = max(total_energy.values()) if total_energy else 1
        normalized = {k: v / max_energy for k, v in total_energy.items()}

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in normalized.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)
