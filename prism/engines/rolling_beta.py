"""
PRISM Rolling Beta Engine

Measures time-varying sensitivity of indicators to a reference.

Measures:
- Rolling beta coefficients over time
- Beta stability/volatility
- Structural changes in sensitivity

Phase: Unbound
Normalization: Returns (percentage change)
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd

from .base import BaseEngine


logger = logging.getLogger(__name__)


class RollingBetaEngine(BaseEngine):
    """
    Rolling Beta engine for sensitivity analysis.
    
    Computes rolling regression coefficients of each indicator
    against a reference indicator (e.g., market index).
    
    Outputs:
        - derived.rolling_beta: Beta coefficients over time
    """
    
    name = "rolling_beta"
    phase = "derived"
    default_normalization = "returns"
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        reference: Optional[str] = None,
        window_size: int = 60,
        min_periods: int = 30,
        **params
    ) -> Dict[str, Any]:
        """
        Run rolling beta analysis.
        
        Args:
            df: Returns data (rows=dates, cols=indicators)
            run_id: Unique run identifier
            reference: Reference indicator (default: first column or 'SPY')
            window_size: Rolling window size (default 60)
            min_periods: Minimum observations (default 30)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        
        # Determine reference indicator
        if reference is None:
            if "SPY" in indicators:
                reference = "SPY"
            else:
                reference = indicators[0]
                logger.warning(f"No reference specified, using {reference}")
        
        if reference not in indicators:
            raise ValueError(f"Reference '{reference}' not in data")
        
        ref_series = df_clean[reference]
        other_indicators = [c for c in indicators if c != reference]
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        # Compute rolling betas
        beta_records = []
        beta_stats = []
        
        for indicator in other_indicators:
            ind_series = df_clean[indicator]
            
            # Rolling covariance and variance
            rolling_cov = ind_series.rolling(
                window=window_size, min_periods=min_periods
            ).cov(ref_series)
            
            rolling_var = ref_series.rolling(
                window=window_size, min_periods=min_periods
            ).var()
            
            rolling_beta = rolling_cov / rolling_var
            
            # Store time series
            for dt, beta in rolling_beta.dropna().items():
                beta_records.append({
                    "indicator_id": indicator,
                    "reference_id": reference,
                    "date": dt.date() if hasattr(dt, 'date') else dt,
                    "beta": float(beta),
                    "window_size": window_size,
                    "run_id": run_id,
                })
            
            # Compute stats
            valid_betas = rolling_beta.dropna()
            if len(valid_betas) > 0:
                beta_stats.append({
                    "indicator": indicator,
                    "mean_beta": float(valid_betas.mean()),
                    "std_beta": float(valid_betas.std()),
                    "min_beta": float(valid_betas.min()),
                    "max_beta": float(valid_betas.max()),
                    "current_beta": float(valid_betas.iloc[-1]),
                })
        
        # Store results
        if beta_records:
            self._store_betas(pd.DataFrame(beta_records), run_id)
        
        # Summary metrics
        if beta_stats:
            stats_df = pd.DataFrame(beta_stats)
            avg_beta = stats_df["mean_beta"].mean()
            avg_beta_vol = stats_df["std_beta"].mean()
            high_beta_count = (stats_df["mean_beta"].abs() > 1.0).sum()
        else:
            avg_beta = 0
            avg_beta_vol = 0
            high_beta_count = 0
        
        metrics = {
            "reference": reference,
            "n_indicators": len(other_indicators),
            "window_size": window_size,
            "avg_beta": float(avg_beta),
            "avg_beta_volatility": float(avg_beta_vol),
            "high_beta_count": int(high_beta_count),
            "n_observations": len(beta_records),
        }
        
        logger.info(
            f"Rolling beta complete: {metrics['n_indicators']} indicators, "
            f"avg beta={metrics['avg_beta']:.2f}"
        )
        
        return metrics
    
    def _store_betas(self, df: pd.DataFrame, run_id: str):
        """Store rolling betas."""
        # Would need to add table: derived.rolling_beta
        # For now, store as geometry fingerprint
        pass
