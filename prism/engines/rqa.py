"""
PRISM RQA (Recurrence Quantification Analysis) Engine

Analyzes system dynamics through recurrence plots.

Measures:
- Recurrence rate (RR)
- Determinism (DET)
- Laminarity (LAM)
- Entropy of diagonal lines
- Average diagonal/vertical line length

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

    Args:
        x: Input time series
        dim: Embedding dimension
        tau: Time delay

    Returns:
        Embedded trajectory matrix (n_vectors x dim)
    """
    n = len(x)
    n_vectors = n - (dim - 1) * tau

    if n_vectors <= 0:
        return np.array([]).reshape(0, dim)

    embedded = np.zeros((n_vectors, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau:i * tau + n_vectors]

    return embedded


def _recurrence_matrix(embedded: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute recurrence matrix.

    R[i,j] = 1 if ||x_i - x_j|| < threshold
    """
    n = embedded.shape[0]
    if n == 0:
        return np.array([]).reshape(0, 0)

    # Compute pairwise distances
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    # Apply threshold
    recurrence = (distances < threshold).astype(int)

    return recurrence


def _diagonal_lines(R: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Extract lengths of diagonal lines from recurrence matrix."""
    n = R.shape[0]
    lines = []

    # Check all diagonals (excluding main diagonal)
    for k in range(1, n):
        diag = np.diag(R, k)

        # Find consecutive 1s
        current_length = 0
        for val in diag:
            if val == 1:
                current_length += 1
            else:
                if current_length >= min_length:
                    lines.append(current_length)
                current_length = 0

        if current_length >= min_length:
            lines.append(current_length)

    return np.array(lines)


def _vertical_lines(R: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Extract lengths of vertical lines from recurrence matrix."""
    n = R.shape[0]
    lines = []

    for j in range(n):
        column = R[:, j]

        current_length = 0
        for val in column:
            if val == 1:
                current_length += 1
            else:
                if current_length >= min_length:
                    lines.append(current_length)
                current_length = 0

        if current_length >= min_length:
            lines.append(current_length)

    return np.array(lines)


def _compute_rqa_metrics(R: np.ndarray, min_line: int = 2) -> Dict[str, float]:
    """
    Compute RQA metrics from recurrence matrix.
    """
    n = R.shape[0]

    if n == 0:
        return {
            "recurrence_rate": 0.0,
            "determinism": 0.0,
            "laminarity": 0.0,
            "avg_diagonal_length": 0.0,
            "avg_vertical_length": 0.0,
            "entropy": 0.0,
            "max_diagonal_length": 0.0,
        }

    # Recurrence rate: fraction of recurrent points
    n_recurrent = R.sum() - n  # Exclude main diagonal
    rr = n_recurrent / (n * (n - 1)) if n > 1 else 0.0

    # Diagonal lines
    diag_lines = _diagonal_lines(R, min_line)

    # Determinism: fraction of recurrent points in diagonal lines
    if n_recurrent > 0 and len(diag_lines) > 0:
        det = diag_lines.sum() / (n_recurrent / 2)  # Divide by 2 for symmetry
        det = min(det, 1.0)  # Cap at 1
    else:
        det = 0.0

    # Average diagonal line length
    avg_diag = diag_lines.mean() if len(diag_lines) > 0 else 0.0
    max_diag = diag_lines.max() if len(diag_lines) > 0 else 0.0

    # Vertical lines (for laminarity)
    vert_lines = _vertical_lines(R, min_line)

    # Laminarity: fraction of recurrent points in vertical lines
    if n_recurrent > 0 and len(vert_lines) > 0:
        lam = vert_lines.sum() / (n_recurrent / 2)
        lam = min(lam, 1.0)
    else:
        lam = 0.0

    avg_vert = vert_lines.mean() if len(vert_lines) > 0 else 0.0

    # Entropy of diagonal line distribution
    if len(diag_lines) > 0:
        hist, _ = np.histogram(diag_lines, bins=range(min_line, int(max_diag) + 2))
        hist = hist[hist > 0]
        if len(hist) > 0:
            p = hist / hist.sum()
            entropy = -np.sum(p * np.log(p))
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    return {
        "recurrence_rate": float(rr),
        "determinism": float(det),
        "laminarity": float(lam),
        "avg_diagonal_length": float(avg_diag),
        "avg_vertical_length": float(avg_vert),
        "entropy": float(entropy),
        "max_diagonal_length": float(max_diag),
    }


class RQAEngine(BaseEngine):
    """
    Recurrence Quantification Analysis engine.

    Analyzes system dynamics through recurrence plots,
    capturing determinism, laminarity, and complexity.

    Outputs:
        - derived.rqa_metrics: Per-indicator RQA metrics
    """

    name = "rqa"
    phase = "derived"
    default_normalization = "zscore"

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        embedding_dim: int = 3,
        time_delay: int = 1,
        threshold_percentile: float = 10.0,
        min_line_length: int = 2,
        **params
    ) -> Dict[str, Any]:
        """
        Run RQA analysis.

        Args:
            df: Normalized indicator data
            run_id: Unique run identifier
            embedding_dim: Embedding dimension (default 3)
            time_delay: Time delay for embedding (default 1)
            threshold_percentile: Percentile for recurrence threshold
            min_line_length: Minimum diagonal/vertical line length

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = df_clean.columns.tolist()
        n_indicators = len(indicators)

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Compute RQA for each indicator
        records = []
        all_rr = []
        all_det = []
        all_lam = []
        all_entropy = []

        for indicator in indicators:
            x = df_clean[indicator].values

            # Embed time series
            embedded = _embed_time_series(x, embedding_dim, time_delay)

            if embedded.shape[0] < 10:
                logger.warning(f"Insufficient data for RQA on {indicator}")
                continue

            # Determine threshold from distance distribution
            diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
            threshold = np.percentile(distances, threshold_percentile)

            # Compute recurrence matrix
            R = _recurrence_matrix(embedded, threshold)

            # Compute RQA metrics
            rqa = _compute_rqa_metrics(R, min_line_length)

            all_rr.append(rqa["recurrence_rate"])
            all_det.append(rqa["determinism"])
            all_lam.append(rqa["laminarity"])
            all_entropy.append(rqa["entropy"])

            records.append({
                "indicator_id": indicator,
                "window_start": window_start,
                "window_end": window_end,
                "recurrence_rate": rqa["recurrence_rate"],
                "determinism": rqa["determinism"],
                "laminarity": rqa["laminarity"],
                "avg_diagonal_length": rqa["avg_diagonal_length"],
                "avg_vertical_length": rqa["avg_vertical_length"],
                "entropy": rqa["entropy"],
                "max_diagonal_length": rqa["max_diagonal_length"],
                "embedding_dim": embedding_dim,
                "time_delay": time_delay,
                "threshold_percentile": threshold_percentile,
                "run_id": run_id,
            })

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("rqa_metrics", df_results, run_id)

        # Summary metrics
        metrics = {
            "n_indicators": n_indicators,
            "n_samples": len(df_clean),
            "embedding_dim": embedding_dim,
            "time_delay": time_delay,
            "threshold_percentile": threshold_percentile,
            "avg_recurrence_rate": float(np.mean(all_rr)) if all_rr else 0.0,
            "avg_determinism": float(np.mean(all_det)) if all_det else 0.0,
            "avg_laminarity": float(np.mean(all_lam)) if all_lam else 0.0,
            "avg_entropy": float(np.mean(all_entropy)) if all_entropy else 0.0,
        }

        logger.info(
            f"RQA complete: {n_indicators} indicators, "
            f"avg_det={metrics['avg_determinism']:.4f}, "
            f"avg_lam={metrics['avg_laminarity']:.4f}"
        )

        return metrics
