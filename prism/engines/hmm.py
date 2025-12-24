"""
PRISM HMM Engine

Hidden Markov Model for probabilistic latent state identification.

Measures:
- Discrete latent states
- State membership probabilities over time
- Transition matrix
- State dwell time statistics

Phase: Structure
Normalization: Z-score preferred
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import date
import warnings

import numpy as np
import pandas as pd

from .base import BaseEngine


logger = logging.getLogger(__name__)


# Try to import hmmlearn, fall back to simple implementation
try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    logger.warning("hmmlearn not installed. Using simplified HMM implementation.")


class HMMEngine(BaseEngine):
    """
    Hidden Markov Model engine for latent state identification.

    Detects discrete latent states from time series behavior.
    Provides state persistence metrics and transition dynamics.

    Outputs:
        - structure.states: State assignments over time
        - structure.hmm_transitions: Transition matrices
    """

    name = "hmm"
    phase = "structure"
    default_normalization = "zscore"

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_states: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 100,
        random_state: int = 42,
        **params
    ) -> Dict[str, Any]:
        """
        Run HMM latent state identification.

        Args:
            df: Normalized indicator data
            run_id: Unique run identifier
            n_states: Number of hidden states (default 3)
            covariance_type: 'diag', 'full', 'spherical'
            n_iter: Max iterations for EM algorithm
            random_state: Random seed for reproducibility

        Returns:
            Dict with summary metrics
        """
        df_clean = df

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Prepare data (samples x features)
        X = df_clean.values

        if HAS_HMMLEARN:
            model, states, probs = self._fit_hmmlearn(
                X, n_states, covariance_type, n_iter, random_state
            )
            transition_matrix = model.transmat_
        else:
            states, probs, transition_matrix = self._fit_simple(
                X, n_states, n_iter, random_state
            )

        # Compute state statistics
        state_statistics = self._compute_state_statistics(states, n_states)

        # Store results
        self._store_state_assignments(
            df_clean.index, states, probs, window_start, window_end, run_id
        )
        self._store_transitions(
            transition_matrix, window_start, window_end, run_id
        )

        # Model selection metrics (AIC/BIC approximation)
        n_params = n_states * (n_states - 1) + n_states * X.shape[1] * 2
        log_likelihood = self._compute_log_likelihood(X, states, probs)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(len(X)) - 2 * log_likelihood

        metrics = {
            "n_states": n_states,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "bic": float(bic),
            "state_statistics": state_statistics,
            "has_hmmlearn": HAS_HMMLEARN,
        }

        logger.info(
            f"HMM complete: {n_states} states, "
            f"BIC={bic:.1f}"
        )

        return metrics

    def _fit_hmmlearn(
        self,
        X: np.ndarray,
        n_states: int,
        covariance_type: str,
        n_iter: int,
        random_state: int
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Fit using hmmlearn library."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state,
            )
            model.fit(X)

            states = model.predict(X)
            probs = model.predict_proba(X)

        return model, states, probs

    def _fit_simple(
        self,
        X: np.ndarray,
        n_states: int,
        n_iter: int,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple K-means based state detection as fallback.
        Not a true HMM, but provides state assignments.
        """
        from sklearn.cluster import KMeans

        np.random.seed(random_state)

        # Use K-means for state assignment
        kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
        states = kmeans.fit_predict(X)

        # Compute soft assignments (distances to centroids)
        distances = kmeans.transform(X)
        # Convert to probabilities (inverse distance weighted)
        inv_distances = 1 / (distances + 1e-10)
        probs = inv_distances / inv_distances.sum(axis=1, keepdims=True)

        # Estimate transition matrix from state sequence
        transition_matrix = self._estimate_transitions(states, n_states)

        return states, probs, transition_matrix

    def _estimate_transitions(
        self,
        states: np.ndarray,
        n_states: int
    ) -> np.ndarray:
        """Estimate transition matrix from state sequence."""
        trans = np.zeros((n_states, n_states))

        for i in range(len(states) - 1):
            trans[states[i], states[i + 1]] += 1

        # Normalize rows
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans = trans / row_sums

        return trans

    def _compute_state_statistics(
        self,
        states: np.ndarray,
        n_states: int
    ) -> Dict[str, Any]:
        """Compute statistics for each latent state."""
        stats = {}

        for state in range(n_states):
            mask = states == state

            # Dwell time analysis (duration in state)
            dwell_times = []
            current_dwell = 0
            for s in states:
                if s == state:
                    current_dwell += 1
                elif current_dwell > 0:
                    dwell_times.append(current_dwell)
                    current_dwell = 0
            if current_dwell > 0:
                dwell_times.append(current_dwell)

            stats[f"state_{state}"] = {
                "state_id": state,  # Numeric only, no labels
                "count": int(mask.sum()),
                "proportion": float(mask.mean()),
                "mean_dwell_time": float(np.mean(dwell_times)) if dwell_times else 0,
                "n_episodes": len(dwell_times),
            }

        return stats

    def _compute_log_likelihood(
        self,
        X: np.ndarray,
        states: np.ndarray,
        probs: np.ndarray
    ) -> float:
        """Approximate log-likelihood."""
        # Use max probability as approximation
        max_probs = probs.max(axis=1)
        return float(np.sum(np.log(max_probs + 1e-10)))

    def _store_state_assignments(
        self,
        index: pd.DatetimeIndex,
        states: np.ndarray,
        probs: np.ndarray,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store state assignments to structure.states."""
        records = []

        # Store state assignment per time point (numeric only)
        for i, (dt, state) in enumerate(zip(index, states)):
            records.append({
                "state_id": int(state),  # Numeric only
                "window_start": window_start,
                "window_end": window_end,
                "confidence": float(probs[i, state]),
                "run_id": run_id,
            })

        # Store summary (one per state)
        unique_states = np.unique(states)
        state_records = []
        for state in unique_states:
            mask = states == state
            state_records.append({
                "state_id": int(state),  # Numeric only
                "window_start": window_start,
                "window_end": window_end,
                "confidence": float(probs[mask, state].mean()),
                "run_id": run_id,
            })

        if state_records:
            df = pd.DataFrame(state_records)
            self.store_results("states", df, run_id)

    def _store_transitions(
        self,
        transition_matrix: np.ndarray,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store transition matrix."""
        # Would need table: structure.hmm_transitions
        # For now, log it
        logger.debug(f"Transition matrix:\n{transition_matrix}")
