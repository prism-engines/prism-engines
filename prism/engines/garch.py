"""
PRISM GARCH Engine

Conditional volatility modeling.

Measures:
- GARCH parameters (persistence)
- Conditional variance series
- Volatility clustering

Phase: Unbound
Normalization: Returns
"""

import logging
from typing import Dict, Any, Optional
from datetime import date
import warnings

import numpy as np
import pandas as pd

from .base import BaseEngine


logger = logging.getLogger(__name__)


# Try to import arch library
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logger.warning("arch library not installed. GARCH will use simplified estimation.")


class GARCHEngine(BaseEngine):
    """
    GARCH engine for volatility analysis.
    
    Models time-varying volatility and captures volatility clustering.
    
    Outputs:
        - derived.geometry_fingerprints: GARCH parameters
    """
    
    name = "garch"
    phase = "derived"
    default_normalization = "returns"
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        p: int = 1,
        q: int = 1,
        **params
    ) -> Dict[str, Any]:
        """
        Run GARCH analysis.
        
        Args:
            df: Returns data
            run_id: Unique run identifier
            p: GARCH lag order (default 1)
            q: ARCH lag order (default 1)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        results = []
        
        for indicator in indicators:
            returns = df_clean[indicator].values * 100  # Scale for numerical stability
            
            try:
                if HAS_ARCH:
                    garch_result = self._fit_arch_garch(returns, p, q)
                else:
                    garch_result = self._fit_simple_garch(returns, p, q)
                
                results.append({
                    "indicator_id": indicator,
                    **garch_result,
                })
                
            except Exception as e:
                logger.warning(f"GARCH failed for {indicator}: {e}")
                results.append({
                    "indicator_id": indicator,
                    "omega": np.nan,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "persistence": np.nan,
                    "unconditional_vol": np.nan,
                })
        
        # Store results
        self._store_garch(results, window_start, window_end, run_id)
        
        # Summary metrics
        df_results = pd.DataFrame(results)
        valid = df_results.dropna(subset=["persistence"])
        
        metrics = {
            "n_indicators": len(indicators),
            "n_successful": len(valid),
            "p": p,
            "q": q,
            "avg_persistence": float(valid["persistence"].mean()) if len(valid) > 0 else None,
            "high_persistence_count": int((valid["persistence"] > 0.9).sum()) if len(valid) > 0 else 0,
            "has_arch_library": HAS_ARCH,
        }
        
        logger.info(
            f"GARCH complete: {metrics['n_successful']}/{len(indicators)} successful, "
            f"avg persistence={metrics['avg_persistence']:.3f}" if metrics['avg_persistence'] else ""
        )
        
        return metrics
    
    def _fit_arch_garch(
        self,
        returns: np.ndarray,
        p: int,
        q: int
    ) -> Dict[str, float]:
        """Fit GARCH using arch library."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
            result = model.fit(disp='off')
            
            params = result.params
            omega = params.get('omega', 0)
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            
            persistence = alpha + beta
            unconditional_vol = np.sqrt(omega / (1 - persistence)) if persistence < 1 else np.nan
            
            return {
                "omega": float(omega),
                "alpha": float(alpha),
                "beta": float(beta),
                "persistence": float(persistence),
                "unconditional_vol": float(unconditional_vol) if not np.isnan(unconditional_vol) else None,
                "log_likelihood": float(result.loglikelihood),
            }
    
    def _fit_simple_garch(
        self,
        returns: np.ndarray,
        p: int,
        q: int
    ) -> Dict[str, float]:
        """
        Simple GARCH(1,1) estimation via variance targeting.
        Fallback when arch library not available.
        """
        n = len(returns)
        
        # Sample variance
        sample_var = np.var(returns)
        
        # Squared returns
        r2 = returns ** 2
        
        # Simple moment estimation for GARCH(1,1)
        # Assumes alpha + beta = 0.9 (typical)
        persistence = 0.9
        
        # Autocorrelation of squared returns
        acf1 = np.corrcoef(r2[:-1], r2[1:])[0, 1]
        
        # alpha ≈ acf1 * (1 - beta)
        # With constraint alpha + beta = persistence
        alpha = min(max(0.05, acf1 * 0.5), 0.3)
        beta = persistence - alpha
        
        # omega from unconditional variance
        omega = sample_var * (1 - persistence)
        
        unconditional_vol = np.sqrt(sample_var)
        
        return {
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "persistence": float(persistence),
            "unconditional_vol": float(unconditional_vol),
            "log_likelihood": None,  # Not computed in simple version
        }
    
    def _store_garch(
        self,
        results: list,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store GARCH parameters as geometry fingerprints."""
        records = []
        
        for r in results:
            for dim in ["alpha", "beta", "persistence", "unconditional_vol"]:
                value = r.get(dim)
                if value is not None and not np.isnan(value):
                    records.append({
                        "indicator_id": r["indicator_id"],
                        "window_start": window_start,
                        "window_end": window_end,
                        "dimension": f"garch_{dim}",
                        "value": float(value),
                        "run_id": run_id,
                    })
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("geometry_fingerprints", df, run_id)
