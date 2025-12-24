"""
PRISM Granger Causality Engine

Tests whether past values of X improve prediction of Y.

Measures:
- F-statistic and p-value per pair
- Directional causality network
- Optimal lag structure

Phase: Unbound
Normalization: None/Z-score (stationarity required)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseEngine


logger = logging.getLogger(__name__)


class GrangerEngine(BaseEngine):
    """
    Granger Causality engine.
    
    Tests pairwise Granger causality between indicators.
    
    Outputs:
        - derived.granger_causality: Pairwise causality tests
    """
    
    name = "granger"
    phase = "derived"
    default_normalization = None  # Handle stationarity internally
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        max_lag: int = 5,
        significance: float = 0.05,
        ensure_stationary: bool = True,
        **params
    ) -> Dict[str, Any]:
        """
        Run Granger causality analysis.
        
        Args:
            df: Indicator data
            run_id: Unique run identifier
            max_lag: Maximum lag to test (default 5)
            significance: Significance level (default 0.05)
            ensure_stationary: Difference if non-stationary (default True)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        n = len(indicators)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        # Ensure stationarity if requested
        if ensure_stationary:
            df_stationary = self._ensure_stationarity(df_clean)
        else:
            df_stationary = df_clean
        
        # Test all pairs
        results = []
        significant_count = 0
        
        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators):
                if i == j:
                    continue
                
                x = df_stationary[ind1].values
                y = df_stationary[ind2].values
                
                # Test if ind1 Granger-causes ind2
                try:
                    f_stat, p_value, optimal_lag = self._granger_test(
                        x, y, max_lag
                    )
                    
                    is_significant = p_value < significance
                    if is_significant:
                        significant_count += 1
                    
                    results.append({
                        "indicator_from": ind1,
                        "indicator_to": ind2,
                        "window_start": window_start,
                        "window_end": window_end,
                        "lag": int(optimal_lag),
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "run_id": run_id,
                    })
                    
                except Exception as e:
                    logger.debug(f"Granger test failed for {ind1} -> {ind2}: {e}")
        
        # Store results
        if results:
            df_results = pd.DataFrame(results)
            self.store_results("granger_causality", df_results, run_id)
        
        # Bonferroni correction info
        n_tests = n * (n - 1)
        bonferroni_threshold = significance / n_tests if n_tests > 0 else significance
        
        metrics = {
            "n_indicators": n,
            "n_tests": len(results),
            "significant_pairs": significant_count,
            "significance_level": significance,
            "bonferroni_threshold": float(bonferroni_threshold),
            "max_lag_tested": max_lag,
        }
        
        logger.info(
            f"Granger causality complete: {significant_count}/{len(results)} "
            f"significant pairs at p<{significance}"
        )
        
        return metrics
    
    def _ensure_stationarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Difference non-stationary series.
        Uses simple first difference (could use ADF test for rigor).
        """
        # Simple approach: always difference
        # More rigorous: use ADF test per series
        return df.diff().dropna()
    
    def _granger_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int
    ) -> Tuple[float, float, int]:
        """
        Test if x Granger-causes y.
        
        Returns (F-statistic, p-value, optimal_lag)
        """
        best_f = 0
        best_p = 1.0
        best_lag = 1
        
        for lag in range(1, max_lag + 1):
            try:
                f_stat, p_value = self._granger_test_single_lag(x, y, lag)
                
                if f_stat > best_f:
                    best_f = f_stat
                    best_p = p_value
                    best_lag = lag
                    
            except Exception:
                continue
        
        return best_f, best_p, best_lag
    
    def _granger_test_single_lag(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int
    ) -> Tuple[float, float]:
        """
        Granger causality test at a single lag.
        
        Compares:
        - Restricted model: y_t = a0 + a1*y_{t-1} + ... + a_lag*y_{t-lag}
        - Unrestricted model: y_t = a0 + a1*y_{t-1} + ... + b1*x_{t-1} + ... + b_lag*x_{t-lag}
        
        F-test on the additional explanatory power of x.
        """
        n = len(y)
        
        if n <= 2 * lag + 1:
            raise ValueError("Not enough observations for this lag")
        
        # Build lagged matrices
        y_target = y[lag:]
        n_obs = len(y_target)
        
        # Restricted model regressors (only y lags)
        X_restricted = np.column_stack([
            y[lag-i-1:n-i-1] for i in range(lag)
        ])
        X_restricted = np.column_stack([np.ones(n_obs), X_restricted])
        
        # Unrestricted model regressors (y lags + x lags)
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag-i-1:n-i-1] for i in range(lag)]
        ])
        
        # Fit both models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Restricted model
            beta_r = np.linalg.lstsq(X_restricted, y_target, rcond=None)[0]
            resid_r = y_target - X_restricted @ beta_r
            rss_r = np.sum(resid_r ** 2)
            
            # Unrestricted model
            beta_u = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)[0]
            resid_u = y_target - X_unrestricted @ beta_u
            rss_u = np.sum(resid_u ** 2)
        
        # F-test
        df_num = lag  # Number of restrictions
        df_den = n_obs - X_unrestricted.shape[1]
        
        if df_den <= 0 or rss_u <= 0:
            raise ValueError("Invalid degrees of freedom")
        
        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
        
        return f_stat, p_value
