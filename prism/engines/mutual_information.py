"""
PRISM Mutual Information Engine

Measures non-linear dependence between indicators.

Measures:
- Mutual information (bits)
- Normalized mutual information
- Non-linear dependency strength

Phase: Unbound
Normalization: Discretization required
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression

from .base import BaseEngine


logger = logging.getLogger(__name__)


class MutualInformationEngine(BaseEngine):
    """
    Mutual Information engine for non-linear dependence.
    
    Captures dependencies that correlation misses.
    
    Outputs:
        - derived.mutual_information: Pairwise MI values
    """
    
    name = "mutual_information"
    phase = "derived"
    default_normalization = None  # Discretize internally
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_bins: int = 8,
        method: str = "binned",
        **params
    ) -> Dict[str, Any]:
        """
        Run mutual information analysis.
        
        Args:
            df: Indicator data
            run_id: Unique run identifier
            n_bins: Number of bins for discretization
            method: 'binned' or 'knn' (k-nearest neighbors)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        n = len(indicators)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        # Discretize for binned MI
        if method == "binned":
            df_discrete = self._discretize(df_clean, n_bins)
        
        # Compute pairwise MI
        results = []
        mi_matrix = np.zeros((n, n))
        
        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators):
                if i > j:
                    continue
                
                if method == "binned":
                    x = df_discrete[ind1].values
                    y = df_discrete[ind2].values
                    mi = mutual_info_score(x, y)
                    nmi = normalized_mutual_info_score(x, y)
                else:  # knn
                    x = df_clean[ind1].values.reshape(-1, 1)
                    y = df_clean[ind2].values
                    mi = mutual_info_regression(x, y, random_state=42)[0]
                    nmi = mi  # No normalized version for continuous
                
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
                
                if i != j:
                    results.append({
                        "indicator_id_1": ind1,
                        "indicator_id_2": ind2,
                        "window_start": window_start,
                        "window_end": window_end,
                        "mutual_information": float(mi),
                        "normalized_mi": float(nmi),
                        "run_id": run_id,
                    })
        
        # Store results
        if results:
            self._store_mi_results(pd.DataFrame(results), run_id)
        
        # Compare with correlation
        corr_matrix = df_clean.corr().values
        upper_idx = np.triu_indices(n, k=1)
        
        mi_values = mi_matrix[upper_idx]
        corr_values = np.abs(corr_matrix[upper_idx])
        
        # Non-linear excess: cases where MI is high but correlation is low
        if len(mi_values) > 0:
            nonlinear_excess = np.mean(mi_values[corr_values < 0.3])
        else:
            nonlinear_excess = 0
        
        metrics = {
            "n_indicators": n,
            "n_pairs": len(results),
            "avg_mi": float(np.mean(mi_values)) if len(mi_values) > 0 else 0,
            "max_mi": float(np.max(mi_values)) if len(mi_values) > 0 else 0,
            "method": method,
            "n_bins": n_bins if method == "binned" else None,
            "nonlinear_excess": float(nonlinear_excess) if not np.isnan(nonlinear_excess) else 0,
        }
        
        logger.info(
            f"Mutual information complete: {metrics['n_pairs']} pairs, "
            f"avg MI={metrics['avg_mi']:.4f}"
        )
        
        return metrics
    
    def _discretize(self, df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
        """Discretize continuous data using quantile bins."""
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            result[col] = pd.qcut(
                df[col], q=n_bins, labels=False, duplicates="drop"
            )
        return result
    
    def _store_mi_results(self, df: pd.DataFrame, run_id: str):
        """Store MI results."""
        # Would need table: derived.mutual_information
        # For now, skip or store as generic
        pass
