"""
PRISM Hurst Exponent Engine

Measures long-term memory and persistence in time series.

Measures:
- Hurst exponent (H)
  - H > 0.5: Trending/persistent (momentum)
  - H = 0.5: Random walk
  - H < 0.5: Mean-reverting (anti-persistent)

Phase: Unbound
Normalization: None (works on levels)
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd

from .base import BaseEngine


logger = logging.getLogger(__name__)


class HurstEngine(BaseEngine):
    """
    Hurst Exponent engine for persistence analysis.
    
    Computes Hurst exponent using R/S (rescaled range) analysis.
    
    Outputs:
        - derived.geometry_fingerprints: Hurst values per indicator
    """
    
    name = "hurst"
    phase = "derived"
    default_normalization = None  # Works on levels
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        min_window: int = 20,
        max_window: Optional[int] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run Hurst exponent analysis.
        
        Args:
            df: Indicator data (levels, not returns)
            run_id: Unique run identifier
            min_window: Minimum window for R/S analysis
            max_window: Maximum window (default: len/4)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        if max_window is None:
            max_window = len(df_clean) // 4
        
        # Compute Hurst for each indicator
        hurst_results = []
        
        for indicator in indicators:
            series = df_clean[indicator].values
            
            try:
                h = self._compute_hurst_rs(series, min_window, max_window)
                
                # Interpret
                if h > 0.55:
                    behavior = "trending"
                elif h < 0.45:
                    behavior = "mean_reverting"
                else:
                    behavior = "random_walk"
                
                hurst_results.append({
                    "indicator_id": indicator,
                    "hurst": float(h),
                    "behavior": behavior,
                })
                
            except Exception as e:
                logger.warning(f"Hurst calculation failed for {indicator}: {e}")
                hurst_results.append({
                    "indicator_id": indicator,
                    "hurst": np.nan,
                    "behavior": "unknown",
                })
        
        # Store as geometry fingerprints
        self._store_hurst(hurst_results, window_start, window_end, run_id)
        
        # Summary metrics
        valid_results = [r for r in hurst_results if not np.isnan(r["hurst"])]
        if valid_results:
            hurst_values = [r["hurst"] for r in valid_results]
            behaviors = [r["behavior"] for r in valid_results]
            
            metrics = {
                "n_indicators": len(indicators),
                "avg_hurst": float(np.mean(hurst_values)),
                "std_hurst": float(np.std(hurst_values)),
                "trending_count": behaviors.count("trending"),
                "mean_reverting_count": behaviors.count("mean_reverting"),
                "random_walk_count": behaviors.count("random_walk"),
                "min_hurst": float(np.min(hurst_values)),
                "max_hurst": float(np.max(hurst_values)),
            }
        else:
            metrics = {"n_indicators": len(indicators), "error": "all calculations failed"}
        
        logger.info(
            f"Hurst analysis complete: avg H={metrics.get('avg_hurst', 'N/A'):.3f}"
        )
        
        return metrics
    
    def _compute_hurst_rs(
        self,
        series: np.ndarray,
        min_window: int,
        max_window: int
    ) -> float:
        """
        Compute Hurst exponent using Rescaled Range (R/S) analysis.
        
        The Hurst exponent H is estimated from the relationship:
            E[R(n)/S(n)] ~ n^H
        
        where R(n) is the range and S(n) is the standard deviation
        over windows of size n.
        """
        n = len(series)
        
        # Generate window sizes (logarithmically spaced)
        window_sizes = []
        size = min_window
        while size <= min(max_window, n // 2):
            window_sizes.append(size)
            size = int(size * 1.5)
        
        if len(window_sizes) < 3:
            raise ValueError("Not enough window sizes for R/S analysis")
        
        rs_values = []
        
        for window_size in window_sizes:
            rs_list = []
            
            # Divide series into non-overlapping windows
            n_windows = n // window_size
            
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                window = series[start:end]
                
                # Compute mean-adjusted series
                mean = np.mean(window)
                deviations = window - mean
                
                # Cumulative deviations
                cumsum = np.cumsum(deviations)
                
                # Range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation
                S = np.std(window, ddof=1)
                
                if S > 0:
                    rs_list.append(R / S)
            
            if rs_list:
                rs_values.append((window_size, np.mean(rs_list)))
        
        if len(rs_values) < 3:
            raise ValueError("Not enough valid R/S values")
        
        # Linear regression on log-log scale
        log_sizes = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        # Fit line: log(R/S) = H * log(n) + c
        slope, _ = np.polyfit(log_sizes, log_rs, 1)
        
        return slope
    
    def _store_hurst(
        self,
        results: list,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store Hurst values as geometry fingerprints."""
        records = []
        for r in results:
            records.append({
                "indicator_id": r["indicator_id"],
                "window_start": window_start,
                "window_end": window_end,
                "dimension": "hurst",
                "value": r["hurst"] if not np.isnan(r["hurst"]) else None,
                "run_id": run_id,
            })
            records.append({
                "indicator_id": r["indicator_id"],
                "window_start": window_start,
                "window_end": window_end,
                "dimension": "persistence_class",
                "value": {"trending": 1, "random_walk": 0, "mean_reverting": -1}.get(r["behavior"], 0),
                "run_id": run_id,
            })
        
        df = pd.DataFrame(records)
        df = df.dropna(subset=["value"])
        
        if not df.empty:
            self.store_results("geometry_fingerprints", df, run_id)
