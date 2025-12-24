"""
PRISM Cross-Correlation Engine

Measures lead/lag relationships between indicators.

Measures:
- Pairwise correlation at various lags
- Optimal lag per pair
- Lead/lag network structure

Phase: Unbound
Normalization: Z-score preferred
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import date

import numpy as np
import pandas as pd
from scipy import signal

from .base import BaseEngine


logger = logging.getLogger(__name__)


class CrossCorrelationEngine(BaseEngine):
    """
    Cross-correlation engine for lead/lag analysis.
    
    Outputs:
        - derived.correlation_matrix: Pairwise correlations at lag 0
        - derived.cross_correlation: Full lag structure per pair
    """
    
    name = "cross_correlation"
    phase = "derived"
    default_normalization = "zscore"
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        max_lag: int = 20,
        store_full_lags: bool = False,
        **params
    ) -> Dict[str, Any]:
        """
        Run cross-correlation analysis.
        
        Args:
            df: Normalized indicator data
            run_id: Unique run identifier
            max_lag: Maximum lag to compute (default 20)
            store_full_lags: Store full lag structure (default False, just optimal)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        n = len(indicators)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        # Compute lag-0 correlation matrix
        corr_matrix = df_clean.corr()
        self._store_correlation_matrix(corr_matrix, window_start, window_end, run_id)
        
        # Compute cross-correlations with lags
        lag_results = []
        
        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators):
                if i >= j:  # Skip self and duplicates
                    continue
                
                x = df_clean[ind1].values
                y = df_clean[ind2].values
                
                # Compute cross-correlation
                xcorr = self._cross_correlate(x, y, max_lag)
                
                # Find optimal lag
                optimal_lag = xcorr["lag"].iloc[xcorr["correlation"].abs().argmax()]
                optimal_corr = xcorr.loc[xcorr["lag"] == optimal_lag, "correlation"].values[0]
                
                lag_results.append({
                    "indicator_id_1": ind1,
                    "indicator_id_2": ind2,
                    "window_start": window_start,
                    "window_end": window_end,
                    "optimal_lag": int(optimal_lag),
                    "optimal_correlation": float(optimal_corr),
                    "lag_0_correlation": float(corr_matrix.loc[ind1, ind2]),
                    "run_id": run_id,
                })
        
        # Store results
        lag_df = pd.DataFrame(lag_results)
        self._store_lag_results(lag_df, run_id)
        
        # Summary metrics
        if len(lag_results) > 0:
            avg_abs_corr = corr_matrix.abs().values[np.triu_indices(n, k=1)].mean()
            leading_pairs = sum(1 for r in lag_results if r["optimal_lag"] < 0)
            lagging_pairs = sum(1 for r in lag_results if r["optimal_lag"] > 0)
        else:
            avg_abs_corr = 0
            leading_pairs = 0
            lagging_pairs = 0
        
        metrics = {
            "n_indicators": n,
            "n_pairs": len(lag_results),
            "avg_abs_correlation": float(avg_abs_corr),
            "max_correlation": float(corr_matrix.abs().values[np.triu_indices(n, k=1)].max()) if n > 1 else 0,
            "leading_pairs": leading_pairs,
            "lagging_pairs": lagging_pairs,
            "synchronous_pairs": len(lag_results) - leading_pairs - lagging_pairs,
        }
        
        logger.info(
            f"Cross-correlation complete: {metrics['n_pairs']} pairs, "
            f"avg |r|={metrics['avg_abs_correlation']:.3f}"
        )
        
        return metrics
    
    def _cross_correlate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int
    ) -> pd.DataFrame:
        """
        Compute normalized cross-correlation at multiple lags.
        
        Positive lag means x leads y.
        """
        n = len(x)
        lags = range(-max_lag, max_lag + 1)
        correlations = []
        
        for lag in lags:
            if lag < 0:
                # x lags y (y leads)
                x_slice = x[-lag:]
                y_slice = y[:lag]
            elif lag > 0:
                # x leads y
                x_slice = x[:-lag]
                y_slice = y[lag:]
            else:
                x_slice = x
                y_slice = y
            
            if len(x_slice) > 0:
                corr = np.corrcoef(x_slice, y_slice)[0, 1]
            else:
                corr = np.nan
            
            correlations.append({"lag": lag, "correlation": corr})
        
        return pd.DataFrame(correlations)
    
    def _store_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store correlation matrix to derived.correlation_matrix."""
        records = []
        indicators = list(corr_matrix.columns)
        
        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators):
                if i > j:  # Store lower triangle only
                    continue
                records.append({
                    "indicator_id_1": ind1,
                    "indicator_id_2": ind2,
                    "window_start": window_start,
                    "window_end": window_end,
                    "correlation": float(corr_matrix.loc[ind1, ind2]),
                    "run_id": run_id,
                })
        
        df = pd.DataFrame(records)
        self.store_results("correlation_matrix", df, run_id)
    
    def _store_lag_results(self, df: pd.DataFrame, run_id: str):
        """Store lag analysis results."""
        # Store to a dedicated cross-correlation table
        # For now, we'll skip since schema doesn't have this table
        # Could add: derived.cross_correlation_lags
        pass
