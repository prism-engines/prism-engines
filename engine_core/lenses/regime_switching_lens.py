"""
Regime Switching Lens - System state detection

Identifies different system states (regimes) over time.
Domain-agnostic: works for any domain (finance, climate, biology, etc.).
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import time

from .base_lens import BaseLens


class RegimeSwitchingLens(BaseLens):
    """
    Regime Switching Lens: Detect system states (regimes).

    Domain-agnostic state detection using Gaussian Mixture Models.

    Provides:
    - State identification (e.g., stable/unstable, low/high activity)
    - State transition probabilities
    - Indicator behavior per state
    """

    name = "regime"
    description = "System state detection and analysis"
    category = "advanced"

    def analyze(
        self,
        df: pd.DataFrame,
        n_regimes: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Detect and analyze system states (regimes).

        Args:
            df: Input DataFrame
            n_regimes: Number of states to detect

        Returns:
            Dictionary with state analysis results
        """
        start_time = time.time()

        self.validate_input(df)
        data = self.prepare_data(df)
        value_cols = self.get_value_columns(data)

        # Scale data
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(data[value_cols])

        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            random_state=42,
            n_init=5
        )
        regime_labels = gmm.fit_predict(scaled_values)

        # Add regime labels to data
        data_with_regime = data.copy()
        data_with_regime['regime'] = regime_labels

        # Regime statistics
        regime_stats = {}
        for regime in range(n_regimes):
            regime_data = data_with_regime[data_with_regime['regime'] == regime]

            if len(regime_data) > 0:
                stats = {
                    "count": int(len(regime_data)),
                    "pct": float(len(regime_data) / len(data) * 100),
                    "mean_values": {},
                    "std_values": {},
                }

                for col in value_cols:
                    stats["mean_values"][col] = float(regime_data[col].mean())
                    stats["std_values"][col] = float(regime_data[col].std())

                # Characterize regime
                overall_mean = np.mean(list(stats["mean_values"].values()))
                overall_std = np.mean(list(stats["std_values"].values()))

                if overall_mean > 0.5:
                    character = "expansion"
                elif overall_mean < -0.5:
                    character = "contraction"
                else:
                    character = "neutral"

                if overall_std > 1.0:
                    character += "_volatile"
                else:
                    character += "_stable"

                stats["character"] = character
                regime_stats[regime] = stats

        # Transition matrix
        transitions = np.zeros((n_regimes, n_regimes))
        for i in range(len(regime_labels) - 1):
            transitions[regime_labels[i], regime_labels[i + 1]] += 1

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transitions, row_sums,
            where=row_sums > 0
        )

        # Regime persistence (probability of staying in same regime)
        persistence = {i: float(transition_matrix[i, i]) for i in range(n_regimes)}

        # Current regime
        current_regime = int(regime_labels[-1])

        # Regime duration analysis
        regime_durations = self._compute_durations(regime_labels, n_regimes)

        result = {
            "n_regimes": n_regimes,
            "regime_labels": regime_labels.tolist(),
            "regime_stats": regime_stats,
            "transition_matrix": transition_matrix.tolist(),
            "persistence": persistence,
            "current_regime": current_regime,
            "current_regime_character": regime_stats.get(current_regime, {}).get("character", "unknown"),
            "regime_durations": regime_durations,
            "dates": [str(d) for d in data["date"].tolist()],
        }

        self._computation_time = time.time() - start_time
        self._last_result = result

        return result

    def _compute_durations(
        self,
        labels: np.ndarray,
        n_regimes: int
    ) -> Dict[int, Dict]:
        """Compute regime duration statistics."""
        durations = {i: [] for i in range(n_regimes)}

        current_regime = labels[0]
        current_duration = 1

        for i in range(1, len(labels)):
            if labels[i] == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = labels[i]
                current_duration = 1

        durations[current_regime].append(current_duration)

        # Compute stats
        duration_stats = {}
        for regime, durs in durations.items():
            if durs:
                duration_stats[regime] = {
                    "mean": float(np.mean(durs)),
                    "max": int(max(durs)),
                    "min": int(min(durs)),
                    "n_episodes": len(durs),
                }
            else:
                duration_stats[regime] = {
                    "mean": 0, "max": 0, "min": 0, "n_episodes": 0
                }

        return duration_stats

    def rank_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Rank indicators by their regime differentiation.

        Indicators with high variance across regimes are more important
        for regime identification.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with indicator rankings
        """
        result = self.analyze(df, **kwargs)
        regime_stats = result["regime_stats"]

        if not regime_stats:
            return pd.DataFrame(columns=["indicator", "score", "rank"])

        # Get all indicators
        first_regime = list(regime_stats.values())[0]
        indicators = list(first_regime.get("mean_values", {}).keys())

        # Compute variance of means across regimes for each indicator
        differentiation = {}
        for indicator in indicators:
            means = []
            for regime_data in regime_stats.values():
                mean_values = regime_data.get("mean_values", {})
                if indicator in mean_values:
                    means.append(mean_values[indicator])

            differentiation[indicator] = float(np.std(means)) if means else 0

        # Normalize
        max_diff = max(differentiation.values()) if differentiation else 1
        normalized = {k: v / max_diff for k, v in differentiation.items()}

        ranking = pd.DataFrame([
            {"indicator": k, "score": v}
            for k, v in normalized.items()
        ])
        ranking = ranking.sort_values("score", ascending=False)
        ranking["rank"] = range(1, len(ranking) + 1)

        return ranking.reset_index(drop=True)
