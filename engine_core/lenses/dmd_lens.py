"""
DMD Lens - Dynamic Mode Decomposition

Extracts dynamic modes from time series data.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import linalg
import time

from .base_lens import BaseLens


class DMDLens(BaseLens):
    """
    DMD Lens: Dynamic Mode Decomposition analysis.

    Provides:
    - Dynamic modes and their frequencies
    - Mode amplitudes and growth rates
    - Indicator participation in each mode
    """

    name = "dmd"
    description = "Dynamic Mode Decomposition for temporal dynamics"
    category = "basic"

    def analyze(
        self,
        df: pd.DataFrame,
        n_modes: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run DMD analysis.

        Args:
            df: Input DataFrame
            n_modes: Number of modes to extract (default: min(n_cols, n_rows/2))

        Returns:
            Dictionary with DMD results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        data = self.normalize_data(data)
        value_cols = self.get_value_columns(data)

        values = data[value_cols].values.T  # Shape: (n_features, n_samples)

        # Build time-delay matrices
        X = values[:, :-1]
        Y = values[:, 1:]

        n_features, n_samples = X.shape
        n_modes = n_modes or min(n_features, n_samples // 2)

        try:
            # SVD of X
            U, S, Vh = linalg.svd(X, full_matrices=False)

            # Truncate to n_modes
            U = U[:, :n_modes]
            S = S[:n_modes]
            Vh = Vh[:n_modes, :]

            # Build reduced matrix
            S_inv = np.diag(1.0 / S)
            A_tilde = U.T @ Y @ Vh.T @ S_inv

            # Eigendecomposition
            eigenvalues, eigenvectors = linalg.eig(A_tilde)

            # DMD modes
            modes = Y @ Vh.T @ S_inv @ eigenvectors

            # Mode frequencies and growth rates
            dt = 1.0  # Assuming unit time step
            frequencies = np.angle(eigenvalues) / (2 * np.pi * dt)
            growth_rates = np.log(np.abs(eigenvalues)) / dt

            # Mode amplitudes
            amplitudes = np.abs(modes).sum(axis=0)

            # Sort by amplitude
            sort_idx = np.argsort(amplitudes)[::-1]
            eigenvalues = eigenvalues[sort_idx]
            modes = modes[:, sort_idx]
            frequencies = frequencies[sort_idx]
            growth_rates = growth_rates[sort_idx]
            amplitudes = amplitudes[sort_idx]

            # Indicator participation in each mode
            mode_participation = {}
            for i in range(min(5, n_modes)):  # Top 5 modes
                mode_loadings = np.abs(modes[:, i])
                mode_loadings = mode_loadings / mode_loadings.sum()
                mode_participation[f"mode_{i+1}"] = {
                    value_cols[j]: float(mode_loadings[j])
                    for j in range(len(value_cols))
                }

            result = {
                "n_modes": n_modes,
                "eigenvalues_real": np.real(eigenvalues).tolist(),
                "eigenvalues_imag": np.imag(eigenvalues).tolist(),
                "frequencies": frequencies.tolist(),
                "growth_rates": growth_rates.tolist(),
                "amplitudes": amplitudes.tolist(),
                "mode_participation": mode_participation,
                "dominant_mode_freq": float(frequencies[0]),
                "dominant_mode_growth": float(growth_rates[0]),
            }

        except Exception as e:
            result = {
                "error": str(e),
                "n_modes": 0,
            }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by their participation in dominant modes.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)

        if "error" in result:
            return pd.DataFrame(columns=["indicator", "score", "rank"])

        mode_participation = result.get("mode_participation", {})

        if not mode_participation:
            return pd.DataFrame(columns=["indicator", "score", "rank"])

        # Average participation across top modes
        all_indicators = set()
        for mode_data in mode_participation.values():
            all_indicators.update(mode_data.keys())

        importance = {}
        for indicator in all_indicators:
            total = 0
            count = 0
            for mode_data in mode_participation.values():
                if indicator in mode_data:
                    total += mode_data[indicator]
                    count += 1
            importance[indicator] = total / count if count > 0 else 0

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in importance.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)
