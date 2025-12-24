"""
PRISM Wavelet Coherence Engine

Time-frequency co-movement analysis.

Measures:
- Coherence at different frequencies/scales
- Phase relationships (lead/lag at each frequency)
- Time-localized correlation

Phase: Unbound
Normalization: None (scale-invariant)
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy import signal

from .base import BaseEngine


logger = logging.getLogger(__name__)


# Try to import pywt for wavelets
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    logger.warning("pywt not installed. Wavelet analysis limited.")


class WaveletEngine(BaseEngine):
    """
    Wavelet Coherence engine for time-frequency analysis.
    
    Captures co-movement at different scales (short-term to long-term).
    
    Outputs:
        - derived.wavelet_coherence: Coherence summaries
    """
    
    name = "wavelet"
    phase = "derived"
    default_normalization = None  # Scale-invariant
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        wavelet: str = "morl",
        scales: Optional[np.ndarray] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run wavelet coherence analysis.
        
        Args:
            df: Indicator data
            run_id: Unique run identifier
            wavelet: Wavelet type ('morl' = Morlet, 'mexh' = Mexican hat)
            scales: Wavelet scales (default: logarithmic range)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        indicators = list(df_clean.columns)
        n = len(indicators)
        n_samples = len(df_clean)
        
        if n_samples < 128:
            logger.warning(
                f"Wavelet analysis works best with 128+ samples. Got {n_samples}."
            )
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        # Default scales (periods from 4 to N/4)
        if scales is None:
            max_scale = n_samples // 4
            scales = np.geomspace(4, max_scale, 20).astype(int)
        
        # Compute pairwise coherence summaries
        results = []
        
        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators):
                if i >= j:
                    continue
                
                x = df_clean[ind1].values
                y = df_clean[ind2].values
                
                # Compute wavelet coherence
                coherence_summary = self._wavelet_coherence(
                    x, y, scales, wavelet
                )
                
                results.append({
                    "indicator_id_1": ind1,
                    "indicator_id_2": ind2,
                    "window_start": window_start,
                    "window_end": window_end,
                    **coherence_summary,
                    "run_id": run_id,
                })
        
        # Store results
        if results:
            self._store_coherence(pd.DataFrame(results), run_id)
        
        # Summary metrics
        df_results = pd.DataFrame(results)
        
        metrics = {
            "n_indicators": n,
            "n_pairs": len(results),
            "n_samples": n_samples,
            "n_scales": len(scales),
            "wavelet": wavelet,
            "avg_short_term_coherence": float(df_results["short_term_coherence"].mean()) if results else 0,
            "avg_long_term_coherence": float(df_results["long_term_coherence"].mean()) if results else 0,
            "has_pywt": HAS_PYWT,
        }
        
        logger.info(
            f"Wavelet analysis complete: {metrics['n_pairs']} pairs, "
            f"short-term coh={metrics['avg_short_term_coherence']:.3f}"
        )
        
        return metrics
    
    def _wavelet_coherence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        scales: np.ndarray,
        wavelet: str
    ) -> Dict[str, float]:
        """
        Compute wavelet coherence summary between two series.
        
        Uses cross-wavelet transform and individual wavelet transforms.
        """
        if HAS_PYWT:
            # Continuous wavelet transform
            cwtx, _ = pywt.cwt(x, scales, wavelet)
            cwty, _ = pywt.cwt(y, scales, wavelet)
        else:
            # Fallback: use scipy's cwt with Ricker wavelet
            cwtx = signal.cwt(x, signal.ricker, scales)
            cwty = signal.cwt(y, signal.ricker, scales)
        
        # Cross-wavelet spectrum
        Wxy = cwtx * np.conj(cwty)
        
        # Individual power spectra
        Wxx = np.abs(cwtx) ** 2
        Wyy = np.abs(cwty) ** 2
        
        # Smoothed coherence (simplified: average over time)
        coherence = np.abs(np.mean(Wxy, axis=1)) ** 2 / (
            np.mean(Wxx, axis=1) * np.mean(Wyy, axis=1) + 1e-10
        )
        
        # Phase (lead/lag)
        phase = np.angle(np.mean(Wxy, axis=1))
        
        # Summarize by scale groups
        n_scales = len(scales)
        short_idx = slice(0, n_scales // 3)  # High frequency (short period)
        mid_idx = slice(n_scales // 3, 2 * n_scales // 3)
        long_idx = slice(2 * n_scales // 3, n_scales)  # Low frequency (long period)
        
        return {
            "short_term_coherence": float(np.mean(coherence[short_idx])),
            "mid_term_coherence": float(np.mean(coherence[mid_idx])),
            "long_term_coherence": float(np.mean(coherence[long_idx])),
            "overall_coherence": float(np.mean(coherence)),
            "dominant_scale": int(scales[np.argmax(coherence)]),
            "short_term_phase": float(np.mean(phase[short_idx])),
            "long_term_phase": float(np.mean(phase[long_idx])),
        }
    
    def _store_coherence(self, df: pd.DataFrame, run_id: str):
        """Store wavelet coherence results."""
        # Would need table: derived.wavelet_coherence
        pass
