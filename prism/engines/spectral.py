"""
PRISM Spectral Density Engine

Frequency domain analysis for dominant cycles.

Measures:
- Power spectral density
- Dominant frequencies/periods
- Spectral peaks

Phase: Unbound
Normalization: Detrend required
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from scipy import signal, fft

from .base import BaseEngine


logger = logging.getLogger(__name__)


class SpectralEngine(BaseEngine):
    """
    Spectral Density engine for cycle detection.
    
    Identifies dominant periodic patterns in time series.
    
    Outputs:
        - derived.geometry_fingerprints: Spectral characteristics
    """
    
    name = "spectral"
    phase = "derived"
    default_normalization = "diff"  # Detrend via differencing
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        method: str = "welch",
        n_peaks: int = 5,
        **params
    ) -> Dict[str, Any]:
        """
        Run spectral analysis.
        
        Args:
            df: Indicator data (preferably detrended)
            run_id: Unique run identifier
            method: 'welch' or 'periodogram'
            n_peaks: Number of spectral peaks to identify
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        n_samples = len(df_clean)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        results = []
        
        for indicator in indicators:
            series = df_clean[indicator].values
            
            # Compute power spectral density
            if method == "welch":
                freqs, psd = signal.welch(series, nperseg=min(256, n_samples // 4))
            else:
                freqs, psd = signal.periodogram(series)
            
            # Find peaks
            peaks_idx, properties = signal.find_peaks(psd, height=np.median(psd))
            
            # Sort by power
            if len(peaks_idx) > 0:
                peak_powers = psd[peaks_idx]
                sorted_idx = np.argsort(peak_powers)[::-1][:n_peaks]
                top_peaks = peaks_idx[sorted_idx]
                
                dominant_freq = freqs[top_peaks[0]] if len(top_peaks) > 0 else 0
                dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf
            else:
                dominant_freq = 0
                dominant_period = np.inf
                top_peaks = []
            
            # Spectral entropy (measure of how spread out the power is)
            psd_norm = psd / (psd.sum() + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            
            results.append({
                "indicator_id": indicator,
                "dominant_frequency": float(dominant_freq),
                "dominant_period": float(dominant_period) if not np.isinf(dominant_period) else None,
                "spectral_entropy": float(spectral_entropy),
                "n_significant_peaks": len(top_peaks),
                "total_power": float(psd.sum()),
            })
        
        # Store as geometry fingerprints
        self._store_spectral(results, window_start, window_end, run_id)
        
        # Summary metrics
        df_results = pd.DataFrame(results)
        
        metrics = {
            "n_indicators": len(indicators),
            "n_samples": n_samples,
            "method": method,
            "avg_spectral_entropy": float(df_results["spectral_entropy"].mean()),
            "avg_n_peaks": float(df_results["n_significant_peaks"].mean()),
        }
        
        logger.info(
            f"Spectral analysis complete: {len(results)} indicators, "
            f"avg entropy={metrics['avg_spectral_entropy']:.2f}"
        )
        
        return metrics
    
    def _store_spectral(
        self,
        results: list,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store spectral characteristics as geometry fingerprints."""
        records = []
        
        for r in results:
            for dim in ["dominant_frequency", "spectral_entropy", "n_significant_peaks"]:
                value = r[dim]
                if value is not None:
                    records.append({
                        "indicator_id": r["indicator_id"],
                        "window_start": window_start,
                        "window_end": window_end,
                        "dimension": dim,
                        "value": float(value),
                        "run_id": run_id,
                    })
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("geometry_fingerprints", df, run_id)
