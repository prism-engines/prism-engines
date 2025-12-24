"""
PRISM Lyapunov Exponent Engine

Estimates largest Lyapunov exponent to detect chaos.

Measures:
- Largest Lyapunov exponent (LLE)
- Indicates sensitivity to initial conditions
- Positive LLE → chaotic behavior
- Negative LLE → stable/convergent

Phase: Unbound
Normalization: Z-score preferred
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd

from .base import BaseEngine


logger = logging.getLogger(__name__)


def _embed_time_series(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Time-delay embedding of a time series.
    """
    n = len(x)
    n_vectors = n - (dim - 1) * tau

    if n_vectors <= 0:
        return np.array([]).reshape(0, dim)

    embedded = np.zeros((n_vectors, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau:i * tau + n_vectors]

    return embedded


def _rosenstein_lyapunov(
    x: np.ndarray,
    dim: int = 3,
    tau: int = 1,
    min_separation: int = 10,
    max_iterations: int = 50
) -> Tuple[float, np.ndarray]:
    """
    Estimate largest Lyapunov exponent using Rosenstein's algorithm.

    Based on: Rosenstein et al. (1993) "A practical method for
    calculating largest Lyapunov exponents from small data sets"

    Returns (lyapunov_exponent, divergence_curve)
    """
    # Embed time series
    embedded = _embed_time_series(x, dim, tau)
    n = embedded.shape[0]

    if n < max_iterations + min_separation:
        return np.nan, np.array([])

    # Find nearest neighbors (excluding temporal neighbors)
    divergence = np.zeros(max_iterations)
    count = np.zeros(max_iterations)

    for i in range(n - max_iterations):
        # Find nearest neighbor at least min_separation apart
        distances = np.linalg.norm(embedded - embedded[i], axis=1)

        # Exclude temporal neighbors
        distances[max(0, i - min_separation):min(n, i + min_separation + 1)] = np.inf

        if np.all(np.isinf(distances)):
            continue

        j = np.argmin(distances)
        initial_dist = distances[j]

        if initial_dist == 0:
            continue

        # Track divergence over time
        for k in range(max_iterations):
            if i + k >= n or j + k >= n:
                break

            dist = np.linalg.norm(embedded[i + k] - embedded[j + k])
            if dist > 0:
                divergence[k] += np.log(dist / initial_dist)
                count[k] += 1

    # Average divergence
    valid = count > 0
    if not np.any(valid):
        return np.nan, np.array([])

    divergence[valid] /= count[valid]

    # Estimate Lyapunov exponent from slope of linear region
    valid_indices = np.where(valid)[0]
    if len(valid_indices) < 5:
        return np.nan, divergence

    # Use first portion of curve (linear region)
    n_fit = min(20, len(valid_indices))
    t = valid_indices[:n_fit]
    d = divergence[t]

    # Linear regression for slope
    slope, _, _, _, _ = np.polyfit(t, d, 1, full=True)
    lyapunov = slope[0] if isinstance(slope, np.ndarray) else slope

    return float(lyapunov), divergence


def _estimate_embedding_params(x: np.ndarray) -> Tuple[int, int]:
    """
    Estimate embedding dimension and time delay.

    Uses autocorrelation for tau and false nearest neighbors for dim.
    Simplified heuristics for robustness.
    """
    n = len(x)

    # Time delay: first zero crossing of autocorrelation
    # or first minimum
    max_lag = min(100, n // 4)
    autocorr = np.correlate(x - x.mean(), x - x.mean(), mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]

    tau = 1
    for i in range(1, max_lag):
        if autocorr[i] < 0.5:  # Threshold approach
            tau = i
            break

    # Embedding dimension: heuristic based on data length
    # Typically 2-10 for financial data
    if n < 200:
        dim = 2
    elif n < 500:
        dim = 3
    else:
        dim = 4

    return dim, tau


class LyapunovEngine(BaseEngine):
    """
    Lyapunov exponent engine.

    Estimates the largest Lyapunov exponent to characterize
    system dynamics (chaos vs stability).

    Interpretation:
    - LLE > 0: Chaotic (exponential divergence)
    - LLE ≈ 0: Marginal stability
    - LLE < 0: Stable (convergent)

    Outputs:
        - derived.lyapunov: Per-indicator Lyapunov estimates
    """

    name = "lyapunov"
    phase = "derived"
    default_normalization = "zscore"

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        embedding_dim: Optional[int] = None,
        time_delay: Optional[int] = None,
        min_separation: int = 10,
        max_iterations: int = 50,
        **params
    ) -> Dict[str, Any]:
        """
        Run Lyapunov exponent estimation.

        Args:
            df: Normalized indicator data
            run_id: Unique run identifier
            embedding_dim: Embedding dimension (auto if None)
            time_delay: Time delay for embedding (auto if None)
            min_separation: Minimum temporal separation for neighbors
            max_iterations: Max iterations for divergence tracking

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = df_clean.columns.tolist()
        n_indicators = len(indicators)

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Compute Lyapunov exponent for each indicator
        records = []
        all_lle = []
        n_chaotic = 0
        n_stable = 0

        for indicator in indicators:
            x = df_clean[indicator].values

            # Estimate embedding parameters if not provided
            if embedding_dim is None or time_delay is None:
                auto_dim, auto_tau = _estimate_embedding_params(x)
                dim = embedding_dim or auto_dim
                tau = time_delay or auto_tau
            else:
                dim, tau = embedding_dim, time_delay

            # Compute Lyapunov exponent
            lle, divergence = _rosenstein_lyapunov(
                x, dim=dim, tau=tau,
                min_separation=min_separation,
                max_iterations=max_iterations
            )

            if not np.isnan(lle):
                all_lle.append(lle)
                if lle > 0.01:
                    n_chaotic += 1
                elif lle < -0.01:
                    n_stable += 1

            records.append({
                "indicator_id": indicator,
                "window_start": window_start,
                "window_end": window_end,
                "lyapunov_exponent": float(lle) if not np.isnan(lle) else None,
                "embedding_dim": dim,
                "time_delay": tau,
                "is_chaotic": lle > 0.01 if not np.isnan(lle) else None,
                "is_stable": lle < -0.01 if not np.isnan(lle) else None,
                "run_id": run_id,
            })

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("lyapunov", df_results, run_id)

        # Summary metrics
        metrics = {
            "n_indicators": n_indicators,
            "n_samples": len(df_clean),
            "n_successful": len(all_lle),
            "avg_lyapunov": float(np.mean(all_lle)) if all_lle else None,
            "max_lyapunov": float(np.max(all_lle)) if all_lle else None,
            "min_lyapunov": float(np.min(all_lle)) if all_lle else None,
            "n_chaotic": n_chaotic,
            "n_stable": n_stable,
            "chaotic_fraction": n_chaotic / len(all_lle) if all_lle else 0.0,
        }

        logger.info(
            f"Lyapunov complete: {n_indicators} indicators, "
            f"avg_LLE={metrics['avg_lyapunov']}, "
            f"chaotic={n_chaotic}, stable={n_stable}"
        )

        return metrics
