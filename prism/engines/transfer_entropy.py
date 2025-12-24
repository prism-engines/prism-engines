"""
PRISM Transfer Entropy Engine

Measures directional information flow between indicators.

Measures:
- Transfer entropy (bits) from X to Y
- Effective transfer entropy (bias-corrected)
- Asymmetry (net information flow direction)

Phase: Unbound
Normalization: Discretization required
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
from collections import Counter

from .base import BaseEngine


logger = logging.getLogger(__name__)


class TransferEntropyEngine(BaseEngine):
    """
    Transfer Entropy engine for information flow analysis.
    
    Measures how much knowing the past of X reduces uncertainty about Y,
    beyond what knowing the past of Y provides.
    
    Outputs:
        - derived.transfer_entropy: Pairwise TE values
    """
    
    name = "transfer_entropy"
    phase = "derived"
    default_normalization = None  # Discretize internally
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_bins: int = 8,
        lag: int = 1,
        n_shuffles: int = 100,
        **params
    ) -> Dict[str, Any]:
        """
        Run transfer entropy analysis.
        
        Args:
            df: Indicator data
            run_id: Unique run identifier
            n_bins: Number of bins for discretization
            lag: Lag for transfer entropy (default 1)
            n_shuffles: Number of shuffles for significance testing
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        n = len(indicators)
        
        if len(df_clean) < 500:
            logger.warning(
                f"Transfer entropy requires substantial data. "
                f"Got {len(df_clean)} samples, recommend 500+"
            )
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        # Discretize
        df_discrete = self._discretize(df_clean, n_bins)
        
        # Compute pairwise TE
        results = []
        
        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators):
                if i == j:
                    continue
                
                x = df_discrete[ind1].values
                y = df_discrete[ind2].values
                
                # TE from ind1 to ind2
                te, te_eff, p_value = self._transfer_entropy(
                    x, y, lag, n_shuffles
                )
                
                results.append({
                    "indicator_from": ind1,
                    "indicator_to": ind2,
                    "window_start": window_start,
                    "window_end": window_end,
                    "transfer_entropy": float(te),
                    "effective_te": float(te_eff),
                    "p_value": float(p_value),
                    "lag": lag,
                    "run_id": run_id,
                })
        
        # Store results
        if results:
            self._store_te_results(pd.DataFrame(results), run_id)
        
        # Find strongest information flows
        df_results = pd.DataFrame(results)
        significant = df_results[df_results["p_value"] < 0.05]
        
        # Net flow analysis
        net_flows = {}
        for ind in indicators:
            outflow = df_results[df_results["indicator_from"] == ind]["effective_te"].sum()
            inflow = df_results[df_results["indicator_to"] == ind]["effective_te"].sum()
            net_flows[ind] = outflow - inflow
        
        # Most influential (net positive flow)
        top_influencer = max(net_flows, key=net_flows.get) if net_flows else None
        
        metrics = {
            "n_indicators": n,
            "n_pairs": len(results),
            "avg_te": float(df_results["transfer_entropy"].mean()),
            "max_te": float(df_results["transfer_entropy"].max()),
            "significant_pairs": len(significant),
            "lag": lag,
            "n_bins": n_bins,
            "top_influencer": top_influencer,
        }
        
        logger.info(
            f"Transfer entropy complete: {metrics['significant_pairs']} "
            f"significant flows out of {metrics['n_pairs']}"
        )
        
        return metrics
    
    def _discretize(self, df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
        """Discretize using quantile bins."""
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            result[col] = pd.qcut(
                df[col], q=n_bins, labels=False, duplicates="drop"
            )
        return result
    
    def _transfer_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
        n_shuffles: int
    ) -> Tuple[float, float, float]:
        """
        Compute transfer entropy from x to y.
        
        TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
        
        Returns (TE, effective_TE, p_value)
        """
        # Build state sequences
        y_past = y[:-lag]
        x_past = x[:-lag]
        y_future = y[lag:]
        
        # Joint and marginal probabilities
        te = self._compute_te(x_past, y_past, y_future)
        
        # Shuffle test for significance
        shuffled_tes = []
        for _ in range(n_shuffles):
            x_shuffled = np.random.permutation(x_past)
            te_shuffled = self._compute_te(x_shuffled, y_past, y_future)
            shuffled_tes.append(te_shuffled)
        
        # Effective TE (bias-corrected)
        bias = np.mean(shuffled_tes)
        te_effective = te - bias
        
        # P-value
        p_value = np.mean([1 if t >= te else 0 for t in shuffled_tes])
        
        return te, max(0, te_effective), p_value
    
    def _compute_te(
        self,
        x_past: np.ndarray,
        y_past: np.ndarray,
        y_future: np.ndarray
    ) -> float:
        """Compute transfer entropy using empirical probabilities."""
        n = len(y_future)
        
        # Count joint occurrences
        # P(y_t, y_{t-1}, x_{t-1})
        joint_xyz = Counter(zip(y_future, y_past, x_past))
        # P(y_{t-1}, x_{t-1})
        joint_xz = Counter(zip(y_past, x_past))
        # P(y_t, y_{t-1})
        joint_yz = Counter(zip(y_future, y_past))
        # P(y_{t-1})
        marginal_z = Counter(y_past)
        
        te = 0.0
        for (yt, yt_1, xt_1), count in joint_xyz.items():
            p_xyz = count / n
            p_xz = joint_xz[(yt_1, xt_1)] / n
            p_yz = joint_yz[(yt, yt_1)] / n
            p_z = marginal_z[yt_1] / n
            
            if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
                te += p_xyz * np.log2((p_xyz * p_z) / (p_xz * p_yz))
        
        return max(0, te)
    
    def _store_te_results(self, df: pd.DataFrame, run_id: str):
        """Store TE results."""
        # Would need table: derived.transfer_entropy
        pass
