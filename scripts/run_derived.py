"""
PRISM Derived Phase: Descriptor-First Orchestrator (Full Engine Suite)

Threading exercise: connects existing components into a unified pipeline.
No resampling. Native series. Native windows. Missing = NULL.

Output contract:
    (indicator_id, window_start, window_end, dimension, value, run_id)

For pairwise engines:
    (indicator_id_1, indicator_id_2, window_start, window_end, dimension, value, run_id)

Usage:
    python run_derived_phase.py
    python run_derived_phase.py --max-indicators 20 --max-windows 3
"""

import duckdb
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import argparse
import logging
import warnings
from scipy import stats, signal
from scipy.optimize import curve_fit
from collections import defaultdict

warnings.filterwarnings('ignore')

# Import PRISM components
import sys
sys.path.insert(0, '/mnt/project')

from prism.db.open import open_prism_db

from scripts.tools.prism_engine_gates import EngineAuditor, EngineStatus, Domain
from scripts.tools.prism_geometry_signature_agent import (
    GeometrySignatureAgent,
    GeometryClass,
    diagnose_saturation,
    diagnose_asymmetry,
    diagnose_bounded_accumulation,
    diagnose_periodicity,
    diagnose_long_memory,
    diagnose_fat_tails,
    diagnose_volatility_clustering,
    diagnose_tail_dependence,
    diagnose_null_dominance,
    diagnose_parameter_instability,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA SETUP - FULL SUITE
# =============================================================================

# Schema is now defined in migrations. See:
#   prism/db/migrations/2025_12_22_derived_schema.sql
#
# Required tables:
#   derived.geometry_descriptors
#   derived.pairwise_descriptors
#   derived.cohort_descriptors
#   derived.regime_assignments
#   derived.wavelet_coefficients
#   derived.correlation_matrix
#   meta.engine_runs
#   meta.engine_skips
#   meta.cohort_membership
REQUIRED_TABLES = [
    "derived.geometry_descriptors",
    "derived.pairwise_descriptors",
    "derived.cohort_descriptors",
    "derived.regime_assignments",
    "derived.wavelet_coefficients",
    "meta.engine_runs",
    "meta.engine_skips",
    "meta.cohort_membership",
]


# =============================================================================
# DESCRIPTOR RESULTS
# =============================================================================

@dataclass
class DescriptorResult:
    """Output from a single-indicator engine."""
    dimension: str
    value: Optional[float]
    
    def is_valid(self) -> bool:
        return self.value is not None and np.isfinite(self.value)


@dataclass
class PairwiseResult:
    """Output from a pairwise engine."""
    indicator_id_1: str
    indicator_id_2: str
    dimension: str
    value: Optional[float]
    p_value: Optional[float] = None
    
    def is_valid(self) -> bool:
        return self.value is not None and np.isfinite(self.value)


@dataclass
class CohortResult:
    """Output from a cohort-level engine."""
    cohort_id: str
    indicator_id: Optional[str]
    dimension: str
    component: Optional[int]
    value: float


@dataclass
class WaveletResult:
    """Output from wavelet engine."""
    scale: int
    frequency_band: str
    energy: float
    dominant_period: Optional[float]


@dataclass
class RegimeResult:
    """Output from regime detection engine."""
    observation_date: str
    regime_id: int
    probability: float


# =============================================================================
# ENGINE ADAPTERS - SINGLE INDICATOR
# =============================================================================

class EngineAdapter:
    """Base class for single-indicator engine adapters."""
    name: str = "base"
    min_observations: int = 30
    engine_type: str = "single"  # single, pairwise, cohort
    
    def can_run(self, n_obs: int) -> bool:
        return n_obs >= self.min_observations
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        raise NotImplementedError


class GeometryDescriptorAdapter(EngineAdapter):
    """
    Wraps all geometry signature diagnostics into descriptor outputs.
    Maps to GeometrySignatureAgent diagnostics.
    """
    name = "geometry_descriptors"
    min_observations = 30
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        # Latent flow diagnostics
        sat = diagnose_saturation(series)
        results.append(DescriptorResult("saturation", sat.score))
        results.append(DescriptorResult("saturation_metric", sat.metric_value))
        
        asym = diagnose_asymmetry(series)
        results.append(DescriptorResult("asymmetry", asym.score))
        
        bounded = diagnose_bounded_accumulation(series)
        results.append(DescriptorResult("bounded_accumulation", bounded.score))
        
        # Oscillator diagnostics
        period = diagnose_periodicity(series)
        results.append(DescriptorResult("periodicity", period.score))
        results.append(DescriptorResult("periodicity_concentration", period.metric_value))
        
        longmem = diagnose_long_memory(series)
        results.append(DescriptorResult("long_memory", longmem.score))
        
        # Reflexive diagnostics
        fat = diagnose_fat_tails(series)
        results.append(DescriptorResult("fat_tails", fat.score))
        results.append(DescriptorResult("excess_kurtosis", fat.metric_value))
        
        volclust = diagnose_volatility_clustering(series)
        results.append(DescriptorResult("volatility_clustering", volclust.score))
        results.append(DescriptorResult("arch_effect", volclust.metric_value))
        
        taildep = diagnose_tail_dependence(series)
        results.append(DescriptorResult("tail_dependence", taildep.score))
        
        # Noise diagnostics
        nulldom = diagnose_null_dominance(series)
        results.append(DescriptorResult("null_dominance", nulldom.score))
        results.append(DescriptorResult("surrogate_zscore", nulldom.metric_value))
        
        paraminstab = diagnose_parameter_instability(series)
        results.append(DescriptorResult("parameter_instability", paraminstab.score))
        
        # Composite scores (from GeometryProfile logic)
        latent_score = np.mean([sat.score, asym.score, bounded.score])
        osc_score = np.mean([period.score, longmem.score])
        reflex_score = np.mean([fat.score, volclust.score, taildep.score])
        noise_score = np.mean([nulldom.score, paraminstab.score])
        
        results.append(DescriptorResult("latent_flow_score", latent_score))
        results.append(DescriptorResult("oscillator_score", osc_score))
        results.append(DescriptorResult("reflexive_score", reflex_score))
        results.append(DescriptorResult("noise_score", noise_score))
        
        # Dominant geometry class
        scores = {
            'latent_flow': latent_score,
            'oscillator': osc_score,
            'reflexive': reflex_score,
            'noise': noise_score
        }
        dominant = max(scores, key=scores.get)
        results.append(DescriptorResult(f"dominant_geometry_{dominant}", 1.0))
        
        return results


class BasicStatsAdapter(EngineAdapter):
    """Basic statistical descriptors on levels and returns."""
    name = "basic_stats"
    min_observations = 10
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        # Level statistics
        results.append(DescriptorResult("level_mean", float(np.mean(series))))
        results.append(DescriptorResult("level_std", float(np.std(series))))
        results.append(DescriptorResult("level_min", float(np.min(series))))
        results.append(DescriptorResult("level_max", float(np.max(series))))
        results.append(DescriptorResult("level_range", float(np.ptp(series))))
        results.append(DescriptorResult("level_median", float(np.median(series))))
        results.append(DescriptorResult("n_observations", float(len(series))))
        
        # Quantiles
        for q in [0.05, 0.25, 0.75, 0.95]:
            results.append(DescriptorResult(f"level_q{int(q*100)}", float(np.percentile(series, q*100))))
        
        # Return statistics (if enough data)
        if len(series) > 1:
            # Simple returns
            returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
            returns = returns[np.isfinite(returns)]
            
            if len(returns) > 0:
                results.append(DescriptorResult("return_mean", float(np.mean(returns))))
                results.append(DescriptorResult("return_std", float(np.std(returns))))
                results.append(DescriptorResult("return_min", float(np.min(returns))))
                results.append(DescriptorResult("return_max", float(np.max(returns))))
                
                if len(returns) > 3:
                    results.append(DescriptorResult("return_skew", float(stats.skew(returns))))
                    results.append(DescriptorResult("return_kurtosis", float(stats.kurtosis(returns))))
                
                # Downside deviation
                neg_returns = returns[returns < 0]
                if len(neg_returns) > 0:
                    results.append(DescriptorResult("downside_std", float(np.std(neg_returns))))
            
            # Log returns
            positive_series = series[series > 0]
            if len(positive_series) > 1:
                log_returns = np.diff(np.log(positive_series))
                log_returns = log_returns[np.isfinite(log_returns)]
                if len(log_returns) > 0:
                    results.append(DescriptorResult("log_return_mean", float(np.mean(log_returns))))
                    results.append(DescriptorResult("log_return_std", float(np.std(log_returns))))
        
        return results


class EntropyAdapter(EngineAdapter):
    """Entropy and complexity measures."""
    name = "entropy"
    min_observations = 50
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        # Permutation entropy (multiple embedding dimensions)
        for order in [3, 4, 5]:
            pe = self._permutation_entropy(series, order=order, delay=1)
            if pe is not None:
                results.append(DescriptorResult(f"permutation_entropy_d{order}", pe))
        
        # Sample entropy
        for m in [2, 3]:
            for r in [0.1, 0.2]:
                se = self._sample_entropy(series, m=m, r=r)
                if se is not None:
                    results.append(DescriptorResult(f"sample_entropy_m{m}_r{int(r*10)}", se))
        
        # Approximate entropy
        apen = self._approximate_entropy(series, m=2, r=0.2)
        if apen is not None:
            results.append(DescriptorResult("approximate_entropy", apen))
        
        # Shannon entropy of histogram
        shannon = self._shannon_entropy(series, bins=20)
        if shannon is not None:
            results.append(DescriptorResult("shannon_entropy", shannon))
        
        # Spectral entropy
        spec_ent = self._spectral_entropy(series)
        if spec_ent is not None:
            results.append(DescriptorResult("spectral_entropy", spec_ent))
        
        return results
    
    def _permutation_entropy(self, series: np.ndarray, order: int = 3, delay: int = 1) -> Optional[float]:
        try:
            n = len(series)
            if n < order * delay + 1:
                return None
            
            patterns = defaultdict(int)
            for i in range(n - (order - 1) * delay):
                window = tuple(series[i + j * delay] for j in range(order))
                pattern = tuple(np.argsort(window))
                patterns[pattern] += 1
            
            total = sum(patterns.values())
            if total == 0:
                return None
            
            probs = [count / total for count in patterns.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            
            max_entropy = np.log2(np.math.factorial(order))
            return entropy / max_entropy if max_entropy > 0 else None
        except:
            return None
    
    def _sample_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> Optional[float]:
        try:
            n = len(series)
            if n < m + 2:
                return None
            
            r_scaled = r * np.std(series)
            if r_scaled == 0:
                return None
            
            def count_matches(template_len):
                count = 0
                for i in range(n - template_len):
                    for j in range(i + 1, n - template_len):
                        if np.max(np.abs(series[i:i+template_len] - series[j:j+template_len])) < r_scaled:
                            count += 1
                return count
            
            A = count_matches(m + 1)
            B = count_matches(m)
            
            if B == 0 or A == 0:
                return None
            
            return -np.log(A / B)
        except:
            return None
    
    def _approximate_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> Optional[float]:
        try:
            n = len(series)
            if n < m + 2:
                return None
            
            r_scaled = r * np.std(series)
            if r_scaled == 0:
                return None
            
            def phi(template_len):
                patterns = []
                for i in range(n - template_len + 1):
                    patterns.append(series[i:i+template_len])
                
                counts = []
                for i, p1 in enumerate(patterns):
                    count = sum(1 for p2 in patterns if np.max(np.abs(p1 - p2)) < r_scaled)
                    counts.append(count / len(patterns))
                
                return np.mean([np.log(c) for c in counts if c > 0])
            
            return phi(m) - phi(m + 1)
        except:
            return None
    
    def _shannon_entropy(self, series: np.ndarray, bins: int = 20) -> Optional[float]:
        try:
            hist, _ = np.histogram(series, bins=bins, density=True)
            hist = hist[hist > 0]
            if len(hist) == 0:
                return None
            return -np.sum(hist * np.log2(hist)) / np.log2(bins)
        except:
            return None
    
    def _spectral_entropy(self, series: np.ndarray) -> Optional[float]:
        try:
            fft = np.fft.fft(series - np.mean(series))
            psd = np.abs(fft[:len(fft)//2])**2
            psd = psd / np.sum(psd)
            psd = psd[psd > 0]
            if len(psd) == 0:
                return None
            return -np.sum(psd * np.log2(psd)) / np.log2(len(psd))
        except:
            return None


class WaveletAdapter(EngineAdapter):
    """Multi-scale wavelet analysis."""
    name = "wavelets"
    min_observations = 64
    
    def compute(self, series: np.ndarray, **kwargs) -> Tuple[List[DescriptorResult], List[WaveletResult]]:
        results = []
        wavelet_results = []
        
        try:
            # Use scipy's cwt with Ricker wavelet
            widths = np.arange(1, min(len(series)//4, 64))
            
            # Continuous wavelet transform
            cwt_matrix = signal.cwt(series - np.mean(series), signal.ricker, widths)
            
            # Energy at each scale
            scale_energies = np.sum(cwt_matrix**2, axis=1)
            total_energy = np.sum(scale_energies)
            
            if total_energy > 0:
                # Normalized energies
                norm_energies = scale_energies / total_energy
                
                # Summary statistics
                results.append(DescriptorResult("wavelet_energy_total", float(total_energy)))
                results.append(DescriptorResult("wavelet_dominant_scale", float(widths[np.argmax(scale_energies)])))
                
                # Energy concentration
                sorted_energies = np.sort(norm_energies)[::-1]
                results.append(DescriptorResult("wavelet_top3_energy", float(np.sum(sorted_energies[:3]))))
                results.append(DescriptorResult("wavelet_entropy", float(-np.sum(norm_energies * np.log2(norm_energies + 1e-10)))))
                
                # Scale bands
                n_scales = len(widths)
                bands = [
                    ("high_freq", 0, n_scales//4),
                    ("mid_freq", n_scales//4, n_scales//2),
                    ("low_freq", n_scales//2, n_scales)
                ]
                
                for band_name, start, end in bands:
                    band_energy = np.sum(norm_energies[start:end])
                    results.append(DescriptorResult(f"wavelet_{band_name}_energy", float(band_energy)))
                    
                    # Wavelet result for detailed table
                    dominant_in_band = widths[start + np.argmax(scale_energies[start:end])]
                    wavelet_results.append(WaveletResult(
                        scale=int(dominant_in_band),
                        frequency_band=band_name,
                        energy=float(band_energy),
                        dominant_period=float(dominant_in_band * 2)  # Approximate period
                    ))
                
                # Wavelet variance (scalogram analysis)
                results.append(DescriptorResult("wavelet_variance", float(np.var(scale_energies))))
                
        except Exception as e:
            logger.debug(f"Wavelet computation failed: {e}")
        
        return results, wavelet_results


class HurstAdapter(EngineAdapter):
    """Hurst exponent estimation (DIAGNOSTIC - long memory)."""
    name = "hurst"
    min_observations = 100
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        # R/S analysis
        hurst_rs = self._hurst_rs(series)
        if hurst_rs is not None:
            results.append(DescriptorResult("hurst_rs", hurst_rs))
            # Interpretation
            if hurst_rs > 0.5:
                results.append(DescriptorResult("hurst_persistent", 1.0))
            elif hurst_rs < 0.5:
                results.append(DescriptorResult("hurst_antipersistent", 1.0))
            else:
                results.append(DescriptorResult("hurst_random", 1.0))
        
        # DFA (Detrended Fluctuation Analysis)
        hurst_dfa = self._hurst_dfa(series)
        if hurst_dfa is not None:
            results.append(DescriptorResult("hurst_dfa", hurst_dfa))
        
        return results
    
    def _hurst_rs(self, series: np.ndarray) -> Optional[float]:
        """Rescaled range (R/S) method."""
        try:
            n = len(series)
            if n < 20:
                return None
            
            # Use multiple window sizes
            sizes = []
            rs_values = []
            
            for size in [int(n/s) for s in [2, 4, 8, 16] if n/s >= 10]:
                n_windows = n // size
                rs_list = []
                
                for i in range(n_windows):
                    window = series[i*size:(i+1)*size]
                    mean_adj = window - np.mean(window)
                    cumsum = np.cumsum(mean_adj)
                    R = np.max(cumsum) - np.min(cumsum)
                    S = np.std(window)
                    if S > 0:
                        rs_list.append(R / S)
                
                if rs_list:
                    sizes.append(size)
                    rs_values.append(np.mean(rs_list))
            
            if len(sizes) < 2:
                return None
            
            # Log-log regression
            log_sizes = np.log(sizes)
            log_rs = np.log(rs_values)
            slope, _ = np.polyfit(log_sizes, log_rs, 1)
            
            return float(slope)
        except:
            return None
    
    def _hurst_dfa(self, series: np.ndarray) -> Optional[float]:
        """Detrended Fluctuation Analysis."""
        try:
            n = len(series)
            if n < 20:
                return None
            
            # Integrate the series
            y = np.cumsum(series - np.mean(series))
            
            scales = []
            fluctuations = []
            
            for scale in [4, 8, 16, 32, 64]:
                if scale > n // 4:
                    break
                
                n_segments = n // scale
                if n_segments < 2:
                    continue
                
                f2 = []
                for i in range(n_segments):
                    segment = y[i*scale:(i+1)*scale]
                    x = np.arange(len(segment))
                    coef = np.polyfit(x, segment, 1)
                    trend = np.polyval(coef, x)
                    f2.append(np.mean((segment - trend)**2))
                
                scales.append(scale)
                fluctuations.append(np.sqrt(np.mean(f2)))
            
            if len(scales) < 2:
                return None
            
            log_scales = np.log(scales)
            log_fluct = np.log(fluctuations)
            slope, _ = np.polyfit(log_scales, log_fluct, 1)
            
            return float(slope)
        except:
            return None


class DMDAdapter(EngineAdapter):
    """Dynamic Mode Decomposition (CONDITIONAL - operator dynamics)."""
    name = "dmd"
    min_observations = 63
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        try:
            # Build time-delay embedding
            delay = 1
            n_modes = min(10, len(series) // 4)
            
            # Create data matrices
            X = np.array([series[i:i+n_modes] for i in range(len(series)-n_modes)]).T
            X1 = X[:, :-1]
            X2 = X[:, 1:]
            
            # SVD of X1
            U, S, Vh = np.linalg.svd(X1, full_matrices=False)
            
            # Truncate to significant modes
            r = min(5, len(S))
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]
            
            # DMD operator
            A_tilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1/S_r)
            
            # Eigenvalues
            eigenvalues = np.linalg.eigvals(A_tilde)
            
            # Descriptors from eigenvalues
            magnitudes = np.abs(eigenvalues)
            phases = np.angle(eigenvalues)
            
            results.append(DescriptorResult("dmd_spectral_radius", float(np.max(magnitudes))))
            results.append(DescriptorResult("dmd_mean_magnitude", float(np.mean(magnitudes))))
            results.append(DescriptorResult("dmd_n_stable_modes", float(np.sum(magnitudes < 1))))
            results.append(DescriptorResult("dmd_n_unstable_modes", float(np.sum(magnitudes > 1))))
            
            # Dominant frequency from phases
            nonzero_phases = phases[np.abs(phases) > 0.01]
            if len(nonzero_phases) > 0:
                results.append(DescriptorResult("dmd_dominant_frequency", float(np.abs(nonzero_phases[0]) / (2 * np.pi))))
            
            # Explained variance from singular values
            total_var = np.sum(S**2)
            if total_var > 0:
                explained = np.sum(S_r**2) / total_var
                results.append(DescriptorResult("dmd_explained_variance", float(explained)))
            
        except Exception as e:
            logger.debug(f"DMD failed: {e}")
        
        return results


class RollingBetaAdapter(EngineAdapter):
    """Rolling beta to benchmark (CONDITIONAL - finance only)."""
    name = "rolling_beta"
    min_observations = 63
    
    def compute(self, series: np.ndarray, benchmark: Optional[np.ndarray] = None, **kwargs) -> List[DescriptorResult]:
        results = []
        
        if benchmark is None or len(benchmark) != len(series):
            # Self-beta (against own lagged returns) as proxy
            if len(series) > 2:
                returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
                returns = returns[np.isfinite(returns)]
                
                if len(returns) > 20:
                    # Rolling autocorrelation as proxy for self-beta
                    window = min(21, len(returns) // 3)
                    betas = []
                    for i in range(window, len(returns)):
                        r1 = returns[i-window:i-1]
                        r2 = returns[i-window+1:i]
                        if np.std(r1) > 0:
                            beta = np.cov(r1, r2)[0, 1] / np.var(r1)
                            if np.isfinite(beta):
                                betas.append(beta)
                    
                    if betas:
                        results.append(DescriptorResult("self_beta_mean", float(np.mean(betas))))
                        results.append(DescriptorResult("self_beta_std", float(np.std(betas))))
            return results
        
        # Actual beta calculation with benchmark
        try:
            returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
            bench_returns = np.diff(benchmark) / (np.abs(benchmark[:-1]) + 1e-10)
            
            # Align
            valid = np.isfinite(returns) & np.isfinite(bench_returns)
            returns = returns[valid]
            bench_returns = bench_returns[valid]
            
            if len(returns) > 20:
                # Overall beta
                cov = np.cov(returns, bench_returns)[0, 1]
                var = np.var(bench_returns)
                if var > 0:
                    beta = cov / var
                    results.append(DescriptorResult("beta", float(beta)))
                
                # Rolling beta
                window = min(21, len(returns) // 3)
                betas = []
                for i in range(window, len(returns)):
                    r = returns[i-window:i]
                    b = bench_returns[i-window:i]
                    cov = np.cov(r, b)[0, 1]
                    var = np.var(b)
                    if var > 0:
                        betas.append(cov / var)
                
                if betas:
                    results.append(DescriptorResult("beta_mean", float(np.mean(betas))))
                    results.append(DescriptorResult("beta_std", float(np.std(betas))))
                    results.append(DescriptorResult("beta_min", float(np.min(betas))))
                    results.append(DescriptorResult("beta_max", float(np.max(betas))))
        except Exception as e:
            logger.debug(f"Rolling beta failed: {e}")
        
        return results


class AutocorrelationAdapter(EngineAdapter):
    """Autocorrelation and partial autocorrelation analysis."""
    name = "autocorrelation"
    min_observations = 50
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        try:
            n = len(series)
            max_lag = min(20, n // 4)
            
            # Demean
            x = series - np.mean(series)
            var = np.var(series)
            
            if var == 0:
                return results
            
            # ACF
            acf_values = []
            for lag in range(1, max_lag + 1):
                if lag < n:
                    acf = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                    if np.isfinite(acf):
                        acf_values.append(acf)
                        results.append(DescriptorResult(f"acf_lag{lag}", float(acf)))
            
            if acf_values:
                # Summary statistics
                results.append(DescriptorResult("acf_mean", float(np.mean(acf_values))))
                results.append(DescriptorResult("acf_decay_rate", self._estimate_decay(acf_values)))
                
                # First zero crossing
                sign_changes = np.where(np.diff(np.sign(acf_values)))[0]
                if len(sign_changes) > 0:
                    results.append(DescriptorResult("acf_first_zero", float(sign_changes[0] + 1)))
                
                # Ljung-Box Q statistic (simplified)
                q_stat = n * (n + 2) * sum((acf**2) / (n - i - 1) for i, acf in enumerate(acf_values[:10]))
                results.append(DescriptorResult("ljung_box_q", float(q_stat)))
                p_value = 1 - stats.chi2.cdf(q_stat, df=min(10, len(acf_values)))
                results.append(DescriptorResult("ljung_box_pvalue", float(p_value)))
            
            # PACF (using Yule-Walker)
            pacf_values = self._compute_pacf(series, max_lag)
            for i, pacf in enumerate(pacf_values):
                results.append(DescriptorResult(f"pacf_lag{i+1}", float(pacf)))
            
            if pacf_values:
                results.append(DescriptorResult("pacf_mean", float(np.mean(np.abs(pacf_values)))))
                
                # Significant PACF lags (> 2/sqrt(n))
                threshold = 2 / np.sqrt(n)
                sig_lags = sum(1 for p in pacf_values if abs(p) > threshold)
                results.append(DescriptorResult("pacf_significant_lags", float(sig_lags)))
            
        except Exception as e:
            logger.debug(f"Autocorrelation failed: {e}")
        
        return results
    
    def _estimate_decay(self, acf_values: List[float]) -> Optional[float]:
        """Estimate exponential decay rate of ACF."""
        try:
            positive_acf = [(i+1, acf) for i, acf in enumerate(acf_values) if acf > 0.01]
            if len(positive_acf) < 3:
                return None
            
            lags, values = zip(*positive_acf)
            log_values = np.log(values)
            slope, _ = np.polyfit(lags, log_values, 1)
            return float(-slope)
        except:
            return None
    
    def _compute_pacf(self, series: np.ndarray, max_lag: int) -> List[float]:
        """Compute PACF using Yule-Walker equations."""
        pacf = []
        try:
            x = series - np.mean(series)
            n = len(x)
            
            for k in range(1, min(max_lag + 1, n // 2)):
                # Build Toeplitz matrix of autocorrelations
                r = [np.corrcoef(x[:-i], x[i:])[0, 1] if i > 0 else 1.0 for i in range(k + 1)]
                r = [ri if np.isfinite(ri) else 0.0 for ri in r]
                
                if k == 1:
                    pacf.append(r[1])
                else:
                    # Solve Yule-Walker
                    R = np.array([[r[abs(i-j)] for j in range(k)] for i in range(k)])
                    r_vec = np.array(r[1:k+1])
                    try:
                        phi = np.linalg.solve(R, r_vec)
                        pacf.append(float(phi[-1]))
                    except:
                        break
        except:
            pass
        
        return pacf


class SpectralAdapter(EngineAdapter):
    """Detailed spectral/FFT analysis (CORE for oscillator detection)."""
    name = "spectral"
    min_observations = 64
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        try:
            n = len(series)
            
            # Detrend
            x = signal.detrend(series)
            
            # FFT
            fft = np.fft.fft(x)
            freqs = np.fft.fftfreq(n)
            
            # Power spectrum (one-sided)
            psd = np.abs(fft[:n//2])**2
            freqs = freqs[:n//2]
            
            total_power = np.sum(psd)
            if total_power == 0:
                return results
            
            # Normalize
            psd_norm = psd / total_power
            
            # Dominant frequency
            peak_idx = np.argmax(psd[1:]) + 1  # Skip DC
            results.append(DescriptorResult("spectral_dominant_freq", float(freqs[peak_idx])))
            results.append(DescriptorResult("spectral_dominant_period", float(1 / freqs[peak_idx]) if freqs[peak_idx] > 0 else None))
            results.append(DescriptorResult("spectral_peak_power", float(psd_norm[peak_idx])))
            
            # Top 3 peaks
            peaks, _ = signal.find_peaks(psd[1:], height=np.max(psd) * 0.1)
            peaks = peaks + 1  # Adjust for skipping DC
            peak_powers = psd[peaks]
            sorted_peaks = peaks[np.argsort(peak_powers)[::-1]]
            
            for i, pk in enumerate(sorted_peaks[:3]):
                results.append(DescriptorResult(f"spectral_peak{i+1}_freq", float(freqs[pk])))
                results.append(DescriptorResult(f"spectral_peak{i+1}_power", float(psd_norm[pk])))
            
            results.append(DescriptorResult("spectral_n_peaks", float(len(peaks))))
            
            # Band powers
            bands = [
                ("ultra_low", 0, 0.05),
                ("low", 0.05, 0.15),
                ("mid", 0.15, 0.30),
                ("high", 0.30, 0.50),
            ]
            
            for band_name, low, high in bands:
                mask = (np.abs(freqs) >= low) & (np.abs(freqs) < high)
                band_power = np.sum(psd_norm[mask])
                results.append(DescriptorResult(f"spectral_{band_name}_power", float(band_power)))
            
            # Spectral centroid (center of mass)
            centroid = np.sum(freqs[1:] * psd_norm[1:]) / np.sum(psd_norm[1:])
            results.append(DescriptorResult("spectral_centroid", float(centroid)))
            
            # Spectral spread (std around centroid)
            spread = np.sqrt(np.sum((freqs[1:] - centroid)**2 * psd_norm[1:]) / np.sum(psd_norm[1:]))
            results.append(DescriptorResult("spectral_spread", float(spread)))
            
            # Spectral flatness (Wiener entropy)
            geo_mean = np.exp(np.mean(np.log(psd_norm[1:] + 1e-10)))
            arith_mean = np.mean(psd_norm[1:])
            flatness = geo_mean / (arith_mean + 1e-10)
            results.append(DescriptorResult("spectral_flatness", float(flatness)))
            
            # Spectral rolloff (frequency below which 85% of power)
            cumsum = np.cumsum(psd_norm)
            rolloff_idx = np.searchsorted(cumsum, 0.85)
            if rolloff_idx < len(freqs):
                results.append(DescriptorResult("spectral_rolloff", float(freqs[rolloff_idx])))
            
        except Exception as e:
            logger.debug(f"Spectral analysis failed: {e}")
        
        return results


class TrendAdapter(EngineAdapter):
    """Trend detection and decomposition."""
    name = "trend"
    min_observations = 30
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        try:
            n = len(series)
            x = np.arange(n)
            
            # Linear trend
            slope, intercept = np.polyfit(x, series, 1)
            results.append(DescriptorResult("trend_slope", float(slope)))
            results.append(DescriptorResult("trend_slope_normalized", float(slope * n / (np.max(series) - np.min(series) + 1e-10))))
            
            # R-squared of linear fit
            predicted = slope * x + intercept
            ss_res = np.sum((series - predicted)**2)
            ss_tot = np.sum((series - np.mean(series))**2)
            r_squared = 1 - ss_res / (ss_tot + 1e-10)
            results.append(DescriptorResult("trend_r_squared", float(r_squared)))
            
            # Quadratic trend
            if n > 10:
                coefs = np.polyfit(x, series, 2)
                results.append(DescriptorResult("trend_quadratic_coef", float(coefs[0])))
                
                # Curvature direction
                if coefs[0] > 0:
                    results.append(DescriptorResult("trend_convex", 1.0))
                else:
                    results.append(DescriptorResult("trend_concave", 1.0))
            
            # Mann-Kendall trend test
            mk_stat, mk_pvalue = self._mann_kendall(series)
            if mk_stat is not None:
                results.append(DescriptorResult("mann_kendall_stat", float(mk_stat)))
                results.append(DescriptorResult("mann_kendall_pvalue", float(mk_pvalue)))
                
                # Trend direction
                if mk_pvalue < 0.05:
                    if mk_stat > 0:
                        results.append(DescriptorResult("trend_increasing", 1.0))
                    else:
                        results.append(DescriptorResult("trend_decreasing", 1.0))
                else:
                    results.append(DescriptorResult("trend_none", 1.0))
            
            # Piecewise linear (breakpoint detection)
            if n > 50:
                breakpoint = self._detect_breakpoint(series)
                if breakpoint is not None:
                    results.append(DescriptorResult("trend_breakpoint", float(breakpoint)))
                    
                    # Slopes before/after
                    slope1, _ = np.polyfit(x[:breakpoint], series[:breakpoint], 1)
                    slope2, _ = np.polyfit(x[breakpoint:], series[breakpoint:], 1)
                    results.append(DescriptorResult("trend_slope_before_break", float(slope1)))
                    results.append(DescriptorResult("trend_slope_after_break", float(slope2)))
                    results.append(DescriptorResult("trend_slope_change", float(slope2 - slope1)))
            
            # Detrended properties
            detrended = series - predicted
            results.append(DescriptorResult("detrended_std", float(np.std(detrended))))
            results.append(DescriptorResult("detrended_range", float(np.ptp(detrended))))
            
        except Exception as e:
            logger.debug(f"Trend analysis failed: {e}")
        
        return results
    
    def _mann_kendall(self, series: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Mann-Kendall trend test."""
        try:
            n = len(series)
            s = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    diff = series[j] - series[i]
                    if diff > 0:
                        s += 1
                    elif diff < 0:
                        s -= 1
            
            # Variance
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            # Z-statistic
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            return float(z), float(p_value)
        except:
            return None, None
    
    def _detect_breakpoint(self, series: np.ndarray) -> Optional[int]:
        """Detect structural breakpoint using residual sum of squares."""
        try:
            n = len(series)
            min_segment = max(10, n // 10)
            
            best_rss = np.inf
            best_break = None
            
            x = np.arange(n)
            
            for bp in range(min_segment, n - min_segment):
                # Fit two segments
                slope1, int1 = np.polyfit(x[:bp], series[:bp], 1)
                slope2, int2 = np.polyfit(x[bp:], series[bp:], 1)
                
                pred1 = slope1 * x[:bp] + int1
                pred2 = slope2 * x[bp:] + int2
                
                rss = np.sum((series[:bp] - pred1)**2) + np.sum((series[bp:] - pred2)**2)
                
                if rss < best_rss:
                    best_rss = rss
                    best_break = bp
            
            # Compare with single line
            slope, intercept = np.polyfit(x, series, 1)
            single_rss = np.sum((series - (slope * x + intercept))**2)
            
            # Only return breakpoint if significantly better
            if best_rss < single_rss * 0.8:
                return best_break
            
            return None
        except:
            return None


class VolatilityAdapter(EngineAdapter):
    """Volatility analysis (GARCH-like features without full model)."""
    name = "volatility"
    min_observations = 63
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        try:
            n = len(series)
            
            # Returns
            if len(series) < 2:
                return results
            
            returns = np.diff(series) / (np.abs(series[:-1]) + 1e-10)
            returns = returns[np.isfinite(returns)]
            
            if len(returns) < 20:
                return results
            
            # Realized volatility
            rv = np.std(returns)
            results.append(DescriptorResult("realized_volatility", float(rv)))
            
            # Annualized (assuming daily)
            results.append(DescriptorResult("annualized_volatility", float(rv * np.sqrt(252))))
            
            # Rolling volatility
            window = min(21, len(returns) // 3)
            rolling_vol = []
            for i in range(window, len(returns)):
                rolling_vol.append(np.std(returns[i-window:i]))
            
            if rolling_vol:
                results.append(DescriptorResult("volatility_mean", float(np.mean(rolling_vol))))
                results.append(DescriptorResult("volatility_std", float(np.std(rolling_vol))))
                results.append(DescriptorResult("volatility_min", float(np.min(rolling_vol))))
                results.append(DescriptorResult("volatility_max", float(np.max(rolling_vol))))
                results.append(DescriptorResult("volatility_range", float(np.max(rolling_vol) - np.min(rolling_vol))))
                
                # Volatility of volatility
                results.append(DescriptorResult("vol_of_vol", float(np.std(rolling_vol) / np.mean(rolling_vol))))
            
            # ARCH effect test (squared returns autocorrelation)
            sq_returns = returns**2
            if len(sq_returns) > 5:
                arch_corr = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1]
                if np.isfinite(arch_corr):
                    results.append(DescriptorResult("arch_autocorr", float(arch_corr)))
            
            # Parkinson volatility (high-low range estimator proxy)
            # Using local max/min as proxy
            local_range = []
            for i in range(0, len(series) - 5, 5):
                window_data = series[i:i+5]
                local_range.append(np.max(window_data) - np.min(window_data))
            
            if local_range:
                parkinson = np.mean(local_range) / (2 * np.sqrt(np.log(2)))
                results.append(DescriptorResult("parkinson_volatility", float(parkinson)))
            
            # Downside volatility
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0:
                results.append(DescriptorResult("downside_volatility", float(np.std(neg_returns))))
                results.append(DescriptorResult("sortino_denominator", float(np.sqrt(np.mean(neg_returns**2)))))
            
            # Upside volatility
            pos_returns = returns[returns > 0]
            if len(pos_returns) > 0:
                results.append(DescriptorResult("upside_volatility", float(np.std(pos_returns))))
            
            # Volatility skew (upside vs downside)
            if len(neg_returns) > 0 and len(pos_returns) > 0:
                vol_skew = np.std(pos_returns) / (np.std(neg_returns) + 1e-10)
                results.append(DescriptorResult("volatility_skew", float(vol_skew)))
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_dd = np.min(drawdowns)
            results.append(DescriptorResult("max_drawdown", float(max_dd)))
            
            # Calmar-like ratio
            if max_dd < 0:
                mean_return = np.mean(returns) * 252
                results.append(DescriptorResult("calmar_ratio", float(mean_return / abs(max_dd))))
            
        except Exception as e:
            logger.debug(f"Volatility analysis failed: {e}")
        
        return results


class RecurrenceAdapter(EngineAdapter):
    """Simplified Recurrence Quantification Analysis (DIAGNOSTIC)."""
    name = "recurrence"
    min_observations = 100
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        try:
            n = len(series)
            
            # Normalize
            x = (series - np.mean(series)) / (np.std(series) + 1e-10)
            
            # Threshold (10% of max distance)
            threshold = 0.1 * np.sqrt(2) * np.std(x)
            
            # Build recurrence matrix (subsampled for efficiency)
            subsample = max(1, n // 100)
            x_sub = x[::subsample]
            n_sub = len(x_sub)
            
            # Distance matrix
            recurrence = np.zeros((n_sub, n_sub))
            for i in range(n_sub):
                for j in range(n_sub):
                    dist = abs(x_sub[i] - x_sub[j])
                    if dist < threshold:
                        recurrence[i, j] = 1
            
            # Recurrence rate
            rr = np.sum(recurrence) / (n_sub * n_sub)
            results.append(DescriptorResult("recurrence_rate", float(rr)))
            
            # Determinism (proportion in diagonal lines)
            det_count = 0
            total_count = 0
            for i in range(-n_sub + 2, n_sub - 1):
                diag = np.diag(recurrence, i)
                # Count lines of length >= 2
                in_line = False
                line_len = 0
                for val in diag:
                    if val == 1:
                        line_len += 1
                        in_line = True
                    else:
                        if in_line and line_len >= 2:
                            det_count += line_len
                        total_count += line_len
                        line_len = 0
                        in_line = False
                if in_line and line_len >= 2:
                    det_count += line_len
                total_count += line_len
            
            if total_count > 0:
                determinism = det_count / total_count
                results.append(DescriptorResult("determinism", float(determinism)))
            
            # Laminarity (vertical lines)
            lam_count = 0
            for j in range(n_sub):
                col = recurrence[:, j]
                in_line = False
                line_len = 0
                for val in col:
                    if val == 1:
                        line_len += 1
                        in_line = True
                    else:
                        if in_line and line_len >= 2:
                            lam_count += line_len
                        line_len = 0
                        in_line = False
                if in_line and line_len >= 2:
                    lam_count += line_len
            
            if n_sub > 0:
                laminarity = lam_count / (n_sub * n_sub)
                results.append(DescriptorResult("laminarity", float(laminarity)))
            
            # Entropy of diagonal line lengths
            line_lengths = []
            for i in range(-n_sub + 2, n_sub - 1):
                diag = np.diag(recurrence, i)
                line_len = 0
                for val in diag:
                    if val == 1:
                        line_len += 1
                    else:
                        if line_len >= 2:
                            line_lengths.append(line_len)
                        line_len = 0
                if line_len >= 2:
                    line_lengths.append(line_len)
            
            if line_lengths:
                # Histogram of line lengths
                unique, counts = np.unique(line_lengths, return_counts=True)
                probs = counts / np.sum(counts)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                results.append(DescriptorResult("recurrence_entropy", float(entropy)))
                results.append(DescriptorResult("mean_diagonal_length", float(np.mean(line_lengths))))
                results.append(DescriptorResult("max_diagonal_length", float(np.max(line_lengths))))
            
        except Exception as e:
            logger.debug(f"Recurrence analysis failed: {e}")
        
        return results


class LyapunovAdapter(EngineAdapter):
    """Simplified Lyapunov exponent estimation (DIAGNOSTIC - chaos detection)."""
    name = "lyapunov"
    min_observations = 200
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        try:
            n = len(series)
            
            # Normalize
            x = (series - np.mean(series)) / (np.std(series) + 1e-10)
            
            # Embedding dimension and delay
            embed_dim = 3
            delay = 1
            
            # Create embedded vectors
            n_vectors = n - (embed_dim - 1) * delay
            if n_vectors < 50:
                return results
            
            vectors = np.array([
                [x[i + j * delay] for j in range(embed_dim)]
                for i in range(n_vectors)
            ])
            
            # Estimate largest Lyapunov exponent using Rosenstein method
            divergences = []
            
            for i in range(min(100, n_vectors - 10)):
                # Find nearest neighbor (not itself or immediate neighbors)
                distances = np.linalg.norm(vectors - vectors[i], axis=1)
                distances[max(0, i-5):min(n_vectors, i+6)] = np.inf
                
                nn_idx = np.argmin(distances)
                
                # Track divergence over time
                for k in range(1, min(10, n_vectors - max(i, nn_idx))):
                    if i + k < n_vectors and nn_idx + k < n_vectors:
                        dist = np.linalg.norm(vectors[i + k] - vectors[nn_idx + k])
                        if dist > 0:
                            divergences.append((k, np.log(dist)))
            
            if divergences:
                # Group by k and average
                k_values = sorted(set(d[0] for d in divergences))
                avg_divergence = []
                for k in k_values:
                    divs = [d[1] for d in divergences if d[0] == k]
                    avg_divergence.append((k, np.mean(divs)))
                
                if len(avg_divergence) > 2:
                    # Fit line to get Lyapunov exponent
                    ks, divs = zip(*avg_divergence)
                    slope, _ = np.polyfit(ks, divs, 1)
                    
                    results.append(DescriptorResult("lyapunov_exponent", float(slope)))
                    
                    # Interpretation
                    if slope > 0.01:
                        results.append(DescriptorResult("chaotic_signature", 1.0))
                    elif slope < -0.01:
                        results.append(DescriptorResult("stable_attractor", 1.0))
                    else:
                        results.append(DescriptorResult("edge_of_chaos", 1.0))
            
            # Correlation dimension estimate
            corr_dim = self._correlation_dimension(vectors)
            if corr_dim is not None:
                results.append(DescriptorResult("correlation_dimension", float(corr_dim)))
            
        except Exception as e:
            logger.debug(f"Lyapunov analysis failed: {e}")
        
        return results
    
    def _correlation_dimension(self, vectors: np.ndarray) -> Optional[float]:
        """Estimate correlation dimension using Grassberger-Procaccia algorithm."""
        try:
            n = len(vectors)
            if n < 20:
                return None
            
            # Subsample for efficiency
            subsample = max(1, n // 50)
            vectors_sub = vectors[::subsample]
            n_sub = len(vectors_sub)
            
            # Compute pairwise distances
            distances = []
            for i in range(n_sub):
                for j in range(i + 1, n_sub):
                    d = np.linalg.norm(vectors_sub[i] - vectors_sub[j])
                    if d > 0:
                        distances.append(d)
            
            if not distances:
                return None
            
            distances = np.array(distances)
            
            # Correlation integral at different scales
            epsilons = np.logspace(np.log10(np.min(distances) + 1e-10), 
                                   np.log10(np.max(distances)), 10)
            
            correlations = []
            for eps in epsilons:
                C = np.sum(distances < eps) / len(distances)
                if C > 0:
                    correlations.append((np.log(eps), np.log(C)))
            
            if len(correlations) < 3:
                return None
            
            log_eps, log_C = zip(*correlations)
            slope, _ = np.polyfit(log_eps, log_C, 1)
            
            return float(slope)
        except:
            return None


class StationarityAdapter(EngineAdapter):
    """Stationarity tests (ADF, KPSS-like)."""
    name = "stationarity"
    min_observations = 50
    
    def compute(self, series: np.ndarray, **kwargs) -> List[DescriptorResult]:
        results = []
        
        try:
            n = len(series)
            
            # Augmented Dickey-Fuller test (simplified)
            adf_stat, adf_pvalue = self._adf_test(series)
            if adf_stat is not None:
                results.append(DescriptorResult("adf_statistic", float(adf_stat)))
                results.append(DescriptorResult("adf_pvalue", float(adf_pvalue)))
                
                # Stationarity decision
                if adf_pvalue < 0.05:
                    results.append(DescriptorResult("is_stationary_adf", 1.0))
                else:
                    results.append(DescriptorResult("is_nonstationary_adf", 1.0))
            
            # Variance ratio test (for random walk)
            vr_stat = self._variance_ratio_test(series)
            if vr_stat is not None:
                results.append(DescriptorResult("variance_ratio", float(vr_stat)))
                
                # Interpretation: VR ≈ 1 suggests random walk
                if abs(vr_stat - 1) < 0.2:
                    results.append(DescriptorResult("random_walk_like", 1.0))
                elif vr_stat > 1.2:
                    results.append(DescriptorResult("positive_autocorr", 1.0))
                else:
                    results.append(DescriptorResult("mean_reverting", 1.0))
            
            # Rolling mean stability
            window = min(n // 4, 50)
            rolling_means = []
            for i in range(0, n - window, window // 2):
                rolling_means.append(np.mean(series[i:i + window]))
            
            if len(rolling_means) > 2:
                mean_stability = np.std(rolling_means) / (np.mean(np.abs(rolling_means)) + 1e-10)
                results.append(DescriptorResult("rolling_mean_stability", float(mean_stability)))
            
            # Rolling variance stability
            rolling_vars = []
            for i in range(0, n - window, window // 2):
                rolling_vars.append(np.var(series[i:i + window]))
            
            if len(rolling_vars) > 2:
                var_stability = np.std(rolling_vars) / (np.mean(rolling_vars) + 1e-10)
                results.append(DescriptorResult("rolling_var_stability", float(var_stability)))
            
            # Half-life of mean reversion (if mean-reverting)
            if adf_pvalue is not None and adf_pvalue < 0.1:
                half_life = self._estimate_half_life(series)
                if half_life is not None and half_life > 0:
                    results.append(DescriptorResult("half_life", float(half_life)))
            
        except Exception as e:
            logger.debug(f"Stationarity test failed: {e}")
        
        return results
    
    def _adf_test(self, series: np.ndarray, max_lag: int = 4) -> Tuple[Optional[float], Optional[float]]:
        """Simplified Augmented Dickey-Fuller test."""
        try:
            n = len(series)
            
            # First difference
            diff = np.diff(series)
            y = diff[max_lag:]
            
            # Build design matrix: lagged level + lagged differences
            X = np.column_stack([
                np.ones(len(y)),
                series[max_lag:-1],  # Lagged level
                *[diff[max_lag-i-1:-i-1] for i in range(max_lag)]  # Lagged differences
            ])
            
            # OLS
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # t-statistic for lagged level coefficient
            residuals = y - X @ beta
            mse = np.sum(residuals**2) / (len(y) - len(beta))
            var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
            
            t_stat = beta[1] / np.sqrt(var_beta[1])
            
            # Approximate p-value using MacKinnon critical values
            # Simplified: use normal approximation (not accurate but indicative)
            p_value = 2 * stats.norm.cdf(t_stat)  # ADF is one-sided, but this is approximate
            
            return float(t_stat), float(p_value)
        except:
            return None, None
    
    def _variance_ratio_test(self, series: np.ndarray, q: int = 4) -> Optional[float]:
        """Variance ratio test (Lo-MacKinlay)."""
        try:
            n = len(series)
            if n < q * 4:
                return None
            
            # Returns
            returns = np.diff(np.log(series[series > 0]))
            if len(returns) < q * 2:
                return None
            
            # Variance of 1-period returns
            var_1 = np.var(returns)
            
            # Variance of q-period returns
            q_returns = np.array([np.sum(returns[i:i+q]) for i in range(len(returns) - q + 1)])
            var_q = np.var(q_returns) / q
            
            if var_1 > 0:
                return float(var_q / var_1)
            
            return None
        except:
            return None
    
    def _estimate_half_life(self, series: np.ndarray) -> Optional[float]:
        """Estimate half-life of mean reversion using AR(1)."""
        try:
            n = len(series)
            
            # AR(1) regression
            y = series[1:]
            x = series[:-1]
            
            slope, _ = np.polyfit(x, y, 1)
            
            # Half-life = -ln(2) / ln(slope)
            if 0 < slope < 1:
                half_life = -np.log(2) / np.log(slope)
                return float(half_life)
            
            return None
        except:
            return None


# =============================================================================
# ENGINE ADAPTERS - PAIRWISE
# =============================================================================

class MutualInformationAdapter(EngineAdapter):
    """Mutual information between indicator pairs (CONDITIONAL)."""
    name = "mutual_information"
    min_observations = 50
    engine_type = "pairwise"
    
    def compute_pairwise(
        self, 
        series1: np.ndarray, 
        series2: np.ndarray,
        indicator_id_1: str,
        indicator_id_2: str
    ) -> List[PairwiseResult]:
        results = []
        
        try:
            # Align by taking common length
            n = min(len(series1), len(series2))
            if n < self.min_observations:
                return results
            
            s1 = series1[:n]
            s2 = series2[:n]
            
            # Discretize for MI calculation
            bins = min(20, int(np.sqrt(n)))
            
            # 2D histogram
            hist_2d, _, _ = np.histogram2d(s1, s2, bins=bins)
            
            # Marginal histograms
            p_xy = hist_2d / np.sum(hist_2d)
            p_x = np.sum(p_xy, axis=1)
            p_y = np.sum(p_xy, axis=0)
            
            # MI calculation
            mi = 0.0
            for i in range(bins):
                for j in range(bins):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            # Normalized MI
            h_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0]))
            h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0]))
            nmi = 2 * mi / (h_x + h_y) if (h_x + h_y) > 0 else 0
            
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "mutual_information", mi))
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "normalized_mi", nmi))
            
            # Permutation test for significance
            n_perms = 100
            mi_null = []
            for _ in range(n_perms):
                s2_perm = np.random.permutation(s2)
                hist_perm, _, _ = np.histogram2d(s1, s2_perm, bins=bins)
                p_perm = hist_perm / np.sum(hist_perm)
                mi_perm = 0.0
                for i in range(bins):
                    for j in range(bins):
                        if p_perm[i, j] > 0 and p_x[i] > 0:
                            p_y_perm = np.sum(p_perm, axis=0)
                            if p_y_perm[j] > 0:
                                mi_perm += p_perm[i, j] * np.log2(p_perm[i, j] / (p_x[i] * p_y_perm[j]))
                mi_null.append(mi_perm)
            
            p_value = np.mean(np.array(mi_null) >= mi)
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "mi_pvalue", mi, p_value))
            
        except Exception as e:
            logger.debug(f"MI computation failed: {e}")
        
        return results


class GrangerAdapter(EngineAdapter):
    """Granger causality (CONDITIONAL - requires stationarity)."""
    name = "granger"
    min_observations = 50
    engine_type = "pairwise"
    
    def compute_pairwise(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        indicator_id_1: str,
        indicator_id_2: str,
        max_lag: int = 4
    ) -> List[PairwiseResult]:
        results = []
        
        try:
            n = min(len(series1), len(series2))
            if n < self.min_observations:
                return results
            
            s1 = series1[:n]
            s2 = series2[:n]
            
            # Test both directions
            for (x, y, dir_name) in [(s1, s2, "1_causes_2"), (s2, s1, "2_causes_1")]:
                f_stat, p_value = self._granger_test(x, y, max_lag)
                if f_stat is not None:
                    results.append(PairwiseResult(
                        indicator_id_1, indicator_id_2, 
                        f"granger_{dir_name}_fstat", f_stat, p_value
                    ))
            
        except Exception as e:
            logger.debug(f"Granger test failed: {e}")
        
        return results
    
    def _granger_test(self, x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[Optional[float], Optional[float]]:
        """Simple Granger causality F-test."""
        try:
            n = len(x)
            if n <= max_lag + 2:
                return None, None
            
            # Restricted model: y ~ y_lags only
            # Unrestricted model: y ~ y_lags + x_lags
            
            # Build design matrices
            Y = y[max_lag:]
            n_obs = len(Y)
            
            # Lagged y
            Y_lags = np.column_stack([y[max_lag-i-1:n-i-1] for i in range(max_lag)])
            
            # Lagged x
            X_lags = np.column_stack([x[max_lag-i-1:n-i-1] for i in range(max_lag)])
            
            # Restricted model
            X_r = np.column_stack([np.ones(n_obs), Y_lags])
            beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
            resid_r = Y - X_r @ beta_r
            ssr_r = np.sum(resid_r**2)
            
            # Unrestricted model
            X_u = np.column_stack([np.ones(n_obs), Y_lags, X_lags])
            beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
            resid_u = Y - X_u @ beta_u
            ssr_u = np.sum(resid_u**2)
            
            # F-statistic
            df_diff = max_lag
            df_resid = n_obs - 2 * max_lag - 1
            
            if ssr_u > 0 and df_resid > 0:
                f_stat = ((ssr_r - ssr_u) / df_diff) / (ssr_u / df_resid)
                p_value = 1 - stats.f.cdf(f_stat, df_diff, df_resid)
                return float(f_stat), float(p_value)
            
            return None, None
        except:
            return None, None


class CopulaAdapter(EngineAdapter):
    """Copula-based tail dependence analysis (CONDITIONAL)."""
    name = "copula"
    min_observations = 100
    engine_type = "pairwise"
    
    def compute_pairwise(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        indicator_id_1: str,
        indicator_id_2: str
    ) -> List[PairwiseResult]:
        results = []
        
        try:
            n = min(len(series1), len(series2))
            if n < self.min_observations:
                return results
            
            s1 = series1[:n]
            s2 = series2[:n]
            
            # Transform to uniform (ranks)
            u1 = stats.rankdata(s1) / (n + 1)
            u2 = stats.rankdata(s2) / (n + 1)
            
            # Kendall's tau
            tau, tau_pvalue = stats.kendalltau(s1, s2)
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "kendall_tau", float(tau), float(tau_pvalue)))
            
            # Spearman's rho
            rho, rho_pvalue = stats.spearmanr(s1, s2)
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "spearman_rho", float(rho), float(rho_pvalue)))
            
            # Upper tail dependence (empirical)
            threshold = 0.9
            upper_mask = (u1 > threshold) & (u2 > threshold)
            if np.sum(upper_mask) > 5:
                upper_tail_dep = np.sum(upper_mask) / np.sum(u1 > threshold)
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "upper_tail_dependence", float(upper_tail_dep)))
            
            # Lower tail dependence (empirical)
            lower_mask = (u1 < (1 - threshold)) & (u2 < (1 - threshold))
            if np.sum(lower_mask) > 5:
                lower_tail_dep = np.sum(lower_mask) / np.sum(u1 < (1 - threshold))
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "lower_tail_dependence", float(lower_tail_dep)))
            
            # Tail asymmetry
            if np.sum(upper_mask) > 5 and np.sum(lower_mask) > 5:
                upper_dep = np.sum(upper_mask) / np.sum(u1 > threshold)
                lower_dep = np.sum(lower_mask) / np.sum(u1 < (1 - threshold))
                tail_asymmetry = upper_dep - lower_dep
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "tail_asymmetry", float(tail_asymmetry)))
            
            # Extremal dependence (joint exceedance probability)
            for q in [0.05, 0.10, 0.95]:
                mask = (u1 > q) & (u2 > q) if q > 0.5 else (u1 < q) & (u2 < q)
                marginal = np.sum(u1 > q if q > 0.5 else u1 < q)
                if marginal > 0:
                    joint_prob = np.sum(mask) / marginal
                    results.append(PairwiseResult(
                        indicator_id_1, indicator_id_2, 
                        f"joint_exceedance_q{int(q*100)}", float(joint_prob)
                    ))
            
            # Gumbel copula parameter estimate (via tau)
            if tau > 0:
                theta_gumbel = 1 / (1 - tau)
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "gumbel_theta", float(theta_gumbel)))
            
            # Clayton copula parameter estimate (via tau)
            if -1 < tau < 1 and tau != 0:
                theta_clayton = 2 * tau / (1 - tau) if tau > 0 else None
                if theta_clayton is not None and theta_clayton > 0:
                    results.append(PairwiseResult(indicator_id_1, indicator_id_2, "clayton_theta", float(theta_clayton)))
            
        except Exception as e:
            logger.debug(f"Copula analysis failed: {e}")
        
        return results


class CointegrationAdapter(EngineAdapter):
    """Cointegration test (DIAGNOSTIC - for pairs trading)."""
    name = "cointegration"
    min_observations = 100
    engine_type = "pairwise"
    
    def compute_pairwise(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        indicator_id_1: str,
        indicator_id_2: str
    ) -> List[PairwiseResult]:
        results = []
        
        try:
            n = min(len(series1), len(series2))
            if n < self.min_observations:
                return results
            
            s1 = series1[:n]
            s2 = series2[:n]
            
            # Engle-Granger two-step cointegration test
            
            # Step 1: OLS regression s1 = a + b*s2 + e
            X = np.column_stack([np.ones(n), s2])
            beta = np.linalg.lstsq(X, s1, rcond=None)[0]
            
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "cointegration_beta", float(beta[1])))
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "cointegration_alpha", float(beta[0])))
            
            # Residuals (spread)
            residuals = s1 - X @ beta
            
            # Step 2: ADF test on residuals
            adf_stat = self._adf_on_residuals(residuals)
            if adf_stat is not None:
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "cointegration_adf_stat", float(adf_stat)))
                
                # Approximate critical values for cointegration (more stringent than regular ADF)
                # 5% critical value approximately -3.37 for 2 variables
                is_cointegrated = 1.0 if adf_stat < -3.37 else 0.0
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "is_cointegrated", is_cointegrated))
            
            # Spread properties
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "spread_mean", float(np.mean(residuals))))
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "spread_std", float(np.std(residuals))))
            
            # Half-life of spread mean reversion
            half_life = self._estimate_half_life(residuals)
            if half_life is not None and half_life > 0:
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "spread_half_life", float(half_life)))
            
            # Spread z-score (current position relative to history)
            z_score = (residuals[-1] - np.mean(residuals)) / (np.std(residuals) + 1e-10)
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "spread_zscore", float(z_score)))
            
            # Hedge ratio stability
            window = n // 4
            hedge_ratios = []
            for i in range(0, n - window, window // 2):
                X_win = np.column_stack([np.ones(window), s2[i:i+window]])
                beta_win = np.linalg.lstsq(X_win, s1[i:i+window], rcond=None)[0]
                hedge_ratios.append(beta_win[1])
            
            if len(hedge_ratios) > 2:
                results.append(PairwiseResult(
                    indicator_id_1, indicator_id_2, 
                    "hedge_ratio_stability", float(np.std(hedge_ratios) / (np.mean(np.abs(hedge_ratios)) + 1e-10))
                ))
            
        except Exception as e:
            logger.debug(f"Cointegration test failed: {e}")
        
        return results
    
    def _adf_on_residuals(self, residuals: np.ndarray, max_lag: int = 4) -> Optional[float]:
        """ADF test on regression residuals."""
        try:
            n = len(residuals)
            
            diff = np.diff(residuals)
            y = diff[max_lag:]
            
            X = np.column_stack([
                residuals[max_lag:-1],
                *[diff[max_lag-i-1:-i-1] for i in range(max_lag)]
            ])
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            resid = y - X @ beta
            mse = np.sum(resid**2) / (len(y) - len(beta))
            var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
            
            t_stat = beta[0] / np.sqrt(var_beta[0])
            return float(t_stat)
        except:
            return None
    
    def _estimate_half_life(self, spread: np.ndarray) -> Optional[float]:
        """Estimate half-life of mean reversion."""
        try:
            y = spread[1:]
            x = spread[:-1]
            slope, _ = np.polyfit(x, y, 1)
            
            if 0 < slope < 1:
                return -np.log(2) / np.log(slope)
            return None
        except:
            return None


class CrossCorrelationAdapter(EngineAdapter):
    """Cross-correlation for lead/lag analysis (CORE)."""
    name = "cross_correlation"
    min_observations = 50
    engine_type = "pairwise"
    
    def compute_pairwise(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        indicator_id_1: str,
        indicator_id_2: str,
        max_lag: int = 20
    ) -> List[PairwiseResult]:
        results = []
        
        try:
            n = min(len(series1), len(series2))
            if n < self.min_observations:
                return results
            
            s1 = series1[:n]
            s2 = series2[:n]
            
            # Normalize
            s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
            s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
            
            max_lag = min(max_lag, n // 4)
            
            # Cross-correlation at different lags
            ccf_values = []
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    ccf = np.corrcoef(s1_norm[:lag], s2_norm[-lag:])[0, 1]
                elif lag > 0:
                    ccf = np.corrcoef(s1_norm[lag:], s2_norm[:-lag])[0, 1]
                else:
                    ccf = np.corrcoef(s1_norm, s2_norm)[0, 1]
                
                if np.isfinite(ccf):
                    ccf_values.append((lag, ccf))
            
            if not ccf_values:
                return results
            
            # Find peak correlation and lag
            lags, ccfs = zip(*ccf_values)
            peak_idx = np.argmax(np.abs(ccfs))
            peak_lag = lags[peak_idx]
            peak_ccf = ccfs[peak_idx]
            
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "peak_correlation", float(peak_ccf)))
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "peak_lag", float(peak_lag)))
            
            # Lead/lag interpretation
            if peak_lag > 0:
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "series1_leads", 1.0))
            elif peak_lag < 0:
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "series2_leads", 1.0))
            else:
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "synchronous", 1.0))
            
            # Zero-lag correlation
            zero_lag_ccf = dict(ccf_values).get(0, 0)
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "zero_lag_correlation", float(zero_lag_ccf)))
            
            # Correlation decay (how fast correlation drops with lag)
            positive_lags = [(l, c) for l, c in ccf_values if l >= 0]
            if len(positive_lags) > 3:
                lags_pos, ccfs_pos = zip(*positive_lags)
                try:
                    decay_rate = -np.polyfit(lags_pos, np.log(np.abs(ccfs_pos) + 1e-10), 1)[0]
                    results.append(PairwiseResult(indicator_id_1, indicator_id_2, "correlation_decay_rate", float(decay_rate)))
                except:
                    pass
            
            # Average absolute correlation
            avg_abs_ccf = np.mean(np.abs(ccfs))
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "avg_abs_correlation", float(avg_abs_ccf)))
            
            # Significance of peak (compared to null)
            threshold = 2 / np.sqrt(n)
            is_significant = 1.0 if abs(peak_ccf) > threshold else 0.0
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "peak_significant", is_significant))
            
        except Exception as e:
            logger.debug(f"Cross-correlation failed: {e}")
        
        return results


class DTWAdapter(EngineAdapter):
    """Dynamic Time Warping similarity (EXCLUDED by gates but useful for comparison)."""
    name = "dtw"
    min_observations = 30
    engine_type = "pairwise"
    
    def compute_pairwise(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        indicator_id_1: str,
        indicator_id_2: str
    ) -> List[PairwiseResult]:
        results = []
        
        try:
            n1, n2 = len(series1), len(series2)
            
            if n1 < self.min_observations or n2 < self.min_observations:
                return results
            
            # Subsample for efficiency if needed
            max_len = 200
            if n1 > max_len:
                step = n1 // max_len
                s1 = series1[::step]
            else:
                s1 = series1
            
            if n2 > max_len:
                step = n2 // max_len
                s2 = series2[::step]
            else:
                s2 = series2
            
            # Normalize
            s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
            s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)
            
            # DTW distance
            dtw_dist, path = self._dtw_distance(s1, s2)
            
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "dtw_distance", float(dtw_dist)))
            
            # Normalized DTW distance
            norm_dist = dtw_dist / (len(s1) + len(s2))
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "dtw_normalized", float(norm_dist)))
            
            # Path characteristics
            if path:
                # Warping degree (deviation from diagonal)
                diagonal_len = np.sqrt(len(s1)**2 + len(s2)**2)
                path_len = len(path)
                warping_degree = path_len / diagonal_len
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "dtw_warping_degree", float(warping_degree)))
                
                # Time offset (average deviation from diagonal)
                offsets = [abs(i - j * len(s1) / len(s2)) for i, j in path]
                avg_offset = np.mean(offsets)
                results.append(PairwiseResult(indicator_id_1, indicator_id_2, "dtw_avg_offset", float(avg_offset)))
            
            # Compare with Euclidean distance
            min_len = min(len(s1), len(s2))
            euclidean = np.sqrt(np.sum((s1[:min_len] - s2[:min_len])**2))
            dtw_ratio = dtw_dist / (euclidean + 1e-10)
            results.append(PairwiseResult(indicator_id_1, indicator_id_2, "dtw_euclidean_ratio", float(dtw_ratio)))
            
        except Exception as e:
            logger.debug(f"DTW failed: {e}")
        
        return results
    
    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute DTW distance and path."""
        n1, n2 = len(s1), len(s2)
        
        # Cost matrix
        dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                cost = (s1[i-1] - s2[j-1])**2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        # Backtrack to find path
        path = []
        i, j = n1, n2
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            if dtw_matrix[i-1, j-1] <= dtw_matrix[i-1, j] and dtw_matrix[i-1, j-1] <= dtw_matrix[i, j-1]:
                i, j = i-1, j-1
            elif dtw_matrix[i-1, j] <= dtw_matrix[i, j-1]:
                i = i - 1
            else:
                j = j - 1
        
        path.reverse()
        
        return np.sqrt(dtw_matrix[n1, n2]), path


# =============================================================================
# ENGINE ADAPTERS - COHORT (PCA, HMM)
# =============================================================================

class PCAAdapter(EngineAdapter):
    """PCA on indicator cohorts (CORE)."""
    name = "pca"
    min_observations = 42
    engine_type = "cohort"
    
    def compute_cohort(
        self,
        data_matrix: np.ndarray,
        indicator_ids: List[str],
        cohort_id: str
    ) -> Tuple[List[CohortResult], List[DescriptorResult]]:
        """
        Compute PCA on cohort data matrix.
        
        Args:
            data_matrix: (n_obs, n_indicators) matrix
            indicator_ids: list of indicator IDs matching columns
            cohort_id: identifier for this cohort
            
        Returns:
            (cohort_results, indicator_results)
        """
        cohort_results = []
        indicator_results = []
        
        try:
            n_obs, n_indicators = data_matrix.shape
            if n_obs < self.min_observations or n_indicators < 2:
                return cohort_results, indicator_results
            
            # Standardize
            means = np.nanmean(data_matrix, axis=0)
            stds = np.nanstd(data_matrix, axis=0)
            stds[stds == 0] = 1
            
            X = (data_matrix - means) / stds
            X = np.nan_to_num(X, 0)
            
            # PCA via SVD
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
            
            # Explained variance
            total_var = np.sum(S**2)
            explained_var = S**2 / total_var if total_var > 0 else S**2
            cumulative_var = np.cumsum(explained_var)
            
            # Number of components to keep
            n_components = min(10, n_indicators, n_obs)
            
            # Cohort-level results
            for i in range(n_components):
                cohort_results.append(CohortResult(
                    cohort_id=cohort_id,
                    indicator_id=None,
                    dimension="pca_variance_explained",
                    component=i+1,
                    value=float(explained_var[i])
                ))
                cohort_results.append(CohortResult(
                    cohort_id=cohort_id,
                    indicator_id=None,
                    dimension="pca_variance_cumulative",
                    component=i+1,
                    value=float(cumulative_var[i])
                ))
            
            # Loadings per indicator
            loadings = Vh.T  # (n_indicators, n_components)
            for j, indicator_id in enumerate(indicator_ids):
                for i in range(min(5, n_components)):  # Top 5 components
                    cohort_results.append(CohortResult(
                        cohort_id=cohort_id,
                        indicator_id=indicator_id,
                        dimension="pca_loading",
                        component=i+1,
                        value=float(loadings[j, i])
                    ))
            
            # Summary descriptors
            indicator_results.append(DescriptorResult("pca_n_components_90pct", 
                float(np.argmax(cumulative_var >= 0.9) + 1) if np.any(cumulative_var >= 0.9) else float(n_components)))
            indicator_results.append(DescriptorResult("pca_pc1_variance", float(explained_var[0])))
            indicator_results.append(DescriptorResult("pca_effective_rank", 
                float(np.exp(-np.sum(explained_var * np.log(explained_var + 1e-10))))))
            
        except Exception as e:
            logger.debug(f"PCA failed: {e}")
        
        return cohort_results, indicator_results


class HMMAdapter(EngineAdapter):
    """Hidden Markov Model for regime detection (CONDITIONAL)."""
    name = "hmm"
    min_observations = 126
    engine_type = "regime"
    max_states = 4
    
    def compute_regimes(
        self,
        series: np.ndarray,
        indicator_id: str,
        dates: Optional[List[str]] = None
    ) -> Tuple[List[DescriptorResult], List[RegimeResult]]:
        """
        Fit HMM and return regime assignments.
        
        Uses a simple Gaussian HMM implementation.
        """
        results = []
        regime_results = []
        
        try:
            n = len(series)
            if n < self.min_observations:
                return results, regime_results
            
            # Try 2, 3, 4 states and pick best BIC
            best_bic = np.inf
            best_states = 2
            best_params = None
            
            for n_states in range(2, self.max_states + 1):
                params, bic = self._fit_hmm(series, n_states)
                if params is not None and bic < best_bic:
                    best_bic = bic
                    best_states = n_states
                    best_params = params
            
            if best_params is None:
                return results, regime_results
            
            means, stds, trans_mat, state_probs = best_params
            
            # Results
            results.append(DescriptorResult("hmm_n_states", float(best_states)))
            results.append(DescriptorResult("hmm_bic", float(best_bic)))
            
            # State characteristics
            for i in range(best_states):
                results.append(DescriptorResult(f"hmm_state{i+1}_mean", float(means[i])))
                results.append(DescriptorResult(f"hmm_state{i+1}_std", float(stds[i])))
                results.append(DescriptorResult(f"hmm_state{i+1}_persistence", float(trans_mat[i, i])))
            
            # Dwell times (expected time in each state)
            for i in range(best_states):
                dwell = 1 / (1 - trans_mat[i, i] + 1e-10)
                results.append(DescriptorResult(f"hmm_state{i+1}_dwell", float(dwell)))
            
            # Regime assignments
            states = self._viterbi(series, means, stds, trans_mat, state_probs)
            
            if dates is not None and len(dates) == len(states):
                for date, state, prob in zip(dates, states, state_probs):
                    regime_results.append(RegimeResult(
                        observation_date=date,
                        regime_id=int(state),
                        probability=float(prob[state]) if len(prob) > state else 0.5
                    ))
            
        except Exception as e:
            logger.debug(f"HMM failed: {e}")
        
        return results, regime_results
    
    def _fit_hmm(self, series: np.ndarray, n_states: int, max_iter: int = 100):
        """Simple Gaussian HMM via EM."""
        try:
            n = len(series)
            
            # Initialize with k-means-like approach
            sorted_data = np.sort(series)
            means = np.array([sorted_data[int(i * n / n_states)] for i in range(n_states)])
            stds = np.ones(n_states) * np.std(series)
            trans_mat = np.ones((n_states, n_states)) / n_states
            np.fill_diagonal(trans_mat, 0.8)
            trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
            state_probs = np.ones(n_states) / n_states
            
            # EM iterations
            for _ in range(max_iter):
                # E-step: compute responsibilities
                log_likes = np.zeros((n, n_states))
                for k in range(n_states):
                    log_likes[:, k] = stats.norm.logpdf(series, means[k], stds[k])
                
                # Forward-backward would be proper, but use simple approach
                responsibilities = np.exp(log_likes - log_likes.max(axis=1, keepdims=True))
                responsibilities /= responsibilities.sum(axis=1, keepdims=True)
                
                # M-step
                for k in range(n_states):
                    weights = responsibilities[:, k]
                    total_weight = weights.sum()
                    if total_weight > 0:
                        means[k] = np.sum(weights * series) / total_weight
                        stds[k] = np.sqrt(np.sum(weights * (series - means[k])**2) / total_weight)
                        stds[k] = max(stds[k], 0.01 * np.std(series))
                
                # Update transitions (simplified)
                for k in range(n_states):
                    trans_mat[k, :] = responsibilities[1:, :].mean(axis=0)
                    trans_mat[k, k] = 0.7 + 0.2 * responsibilities[:-1, k].mean()
                trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
            
            # BIC
            log_like = np.sum(np.log(np.sum(np.exp(log_likes), axis=1)))
            n_params = n_states * 2 + n_states * (n_states - 1)
            bic = -2 * log_like + n_params * np.log(n)
            
            return (means, stds, trans_mat, responsibilities), bic
        except:
            return None, np.inf
    
    def _viterbi(self, series, means, stds, trans_mat, state_probs):
        """Viterbi decoding for most likely state sequence."""
        n = len(series)
        n_states = len(means)
        
        # Log probabilities
        log_trans = np.log(trans_mat + 1e-10)
        log_emit = np.zeros((n, n_states))
        for k in range(n_states):
            log_emit[:, k] = stats.norm.logpdf(series, means[k], stds[k])
        
        # Viterbi
        V = np.zeros((n, n_states))
        path = np.zeros((n, n_states), dtype=int)
        
        V[0] = np.log(state_probs[0] + 1e-10) + log_emit[0]
        
        for t in range(1, n):
            for k in range(n_states):
                probs = V[t-1] + log_trans[:, k]
                path[t, k] = np.argmax(probs)
                V[t, k] = probs[path[t, k]] + log_emit[t, k]
        
        # Backtrack
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(n-2, -1, -1):
            states[t] = path[t+1, states[t+1]]
        
        return states


# =============================================================================
# PERSISTENCE HELPERS
# =============================================================================

def write_descriptors(
    conn: duckdb.DuckDBPyConnection,
    indicator_id: str,
    window_start: str,
    window_end: str,
    descriptors: List[DescriptorResult],
    run_id: str
) -> Tuple[int, int]:
    """Write single-indicator descriptor rows."""
    written = 0
    skipped = 0
    now = datetime.now()
    
    for desc in descriptors:
        if desc.is_valid():
            try:
                conn.execute("""
                    INSERT INTO derived.geometry_descriptors 
                    (indicator_id, window_start, window_end, dimension, value, run_id, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO UPDATE SET value = EXCLUDED.value, computed_at = EXCLUDED.computed_at
                """, [indicator_id, window_start, window_end, desc.dimension, desc.value, run_id, now])
                written += 1
            except Exception as e:
                logger.debug(f"Write failed for {indicator_id}/{desc.dimension}: {e}")
                skipped += 1
        else:
            skipped += 1
    
    return written, skipped


def write_pairwise_descriptors(
    conn: duckdb.DuckDBPyConnection,
    window_start: str,
    window_end: str,
    results: List[PairwiseResult],
    run_id: str
) -> Tuple[int, int]:
    """Write pairwise descriptor rows."""
    written = 0
    skipped = 0
    now = datetime.now()
    
    for res in results:
        if res.is_valid():
            try:
                conn.execute("""
                    INSERT INTO derived.pairwise_descriptors 
                    (indicator_id_1, indicator_id_2, window_start, window_end, dimension, value, p_value, run_id, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO UPDATE SET value = EXCLUDED.value, p_value = EXCLUDED.p_value, computed_at = EXCLUDED.computed_at
                """, [res.indicator_id_1, res.indicator_id_2, window_start, window_end, 
                      res.dimension, res.value, res.p_value, run_id, now])
                written += 1
            except Exception as e:
                logger.debug(f"Pairwise write failed: {e}")
                skipped += 1
        else:
            skipped += 1
    
    return written, skipped


def write_cohort_descriptors(
    conn: duckdb.DuckDBPyConnection,
    window_start: str,
    window_end: str,
    results: List[CohortResult],
    run_id: str
) -> Tuple[int, int]:
    """Write cohort-level descriptor rows."""
    written = 0
    skipped = 0
    now = datetime.now()
    
    for res in results:
        try:
            conn.execute("""
                INSERT INTO derived.cohort_descriptors 
                (cohort_id, window_start, window_end, dimension, value, run_id, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET value = EXCLUDED.value, computed_at = EXCLUDED.computed_at
            """, [res.cohort_id, window_start, window_end,
                  res.dimension, res.value, run_id, now])
            written += 1
        except Exception as e:
            logger.debug(f"Cohort write failed: {e}")
            skipped += 1
    
    return written, skipped


def write_wavelet_results(
    conn: duckdb.DuckDBPyConnection,
    indicator_id: str,
    window_start: str,
    window_end: str,
    results: List[WaveletResult],
    run_id: str
) -> Tuple[int, int]:
    """Write wavelet coefficient rows."""
    written = 0
    skipped = 0
    now = datetime.now()
    
    for res in results:
        try:
            conn.execute("""
                INSERT INTO derived.wavelet_coefficients 
                (indicator_id, window_start, window_end, scale, coefficient_type, value, run_id, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET value = EXCLUDED.value, computed_at = EXCLUDED.computed_at
            """, [indicator_id, window_start, window_end, res.scale, res.frequency_band,
                  res.energy, run_id, now])
            written += 1
        except Exception as e:
            logger.debug(f"Wavelet write failed: {e}")
            skipped += 1
    
    return written, skipped


def write_regime_results(
    conn: duckdb.DuckDBPyConnection,
    indicator_id: str,
    window_start: str,
    window_end: str,
    results: List[RegimeResult],
    run_id: str
) -> Tuple[int, int]:
    """Write regime assignment rows."""
    written = 0
    skipped = 0
    now = datetime.now()
    
    for res in results:
        try:
            conn.execute("""
                INSERT INTO derived.regime_assignments 
                (indicator_id, date, regime, probability, run_id, computed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET regime = EXCLUDED.regime, probability = EXCLUDED.probability, computed_at = EXCLUDED.computed_at
            """, [indicator_id, res.observation_date, res.regime_id, res.probability, run_id, now])
            written += 1
        except Exception as e:
            logger.debug(f"Regime write failed: {e}")
            skipped += 1
    
    return written, skipped


def log_engine_run(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    engine_name: str,
    window_start: str,
    window_end: str,
    status: str,
    rows_written: int,
    rows_skipped: int,
    started_at: datetime,
    parameters: Optional[Dict] = None,
    error_message: Optional[str] = None
):
    """Log engine run to meta.engine_runs."""
    try:
        completed_at = datetime.utcnow()
        conn.execute("""
            INSERT INTO meta.engine_runs
            (run_id, engine_name, phase, started_at, completed_at, status,
             rows_written, rows_skipped, window_start, window_end, parameters, error_message)
            VALUES (?, ?, 'derived', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET
                completed_at = ?,
                status = EXCLUDED.status,
                rows_written = EXCLUDED.rows_written,
                rows_skipped = EXCLUDED.rows_skipped,
                error_message = EXCLUDED.error_message
        """, [
            run_id, engine_name, started_at, completed_at, status, rows_written, rows_skipped,
            window_start, window_end,
            str(parameters) if parameters else None,
            error_message,
            completed_at
        ])
    except Exception as e:
        logger.error(f"Failed to log engine run: {e}")


def log_skip(
    conn: duckdb.DuckDBPyConnection,
    engine_name: str,
    indicator_id: Optional[str],
    window_start: str,
    window_end: str,
    reason: str,
    details: Optional[str] = None
):
    """Log engine skip for audit trail."""
    try:
        skip_id = f"{engine_name}_{indicator_id or 'cohort'}_{window_start}_{uuid.uuid4().hex[:8]}"
        conn.execute("""
            INSERT INTO meta.engine_skips 
            (skip_id, engine_name, indicator_id, window_start, window_end, skip_reason, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [skip_id, engine_name, indicator_id, window_start, window_end, reason, details])
    except:
        pass


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class DerivedPhaseOrchestrator:
    """
    Descriptor-first derived phase orchestrator - FULL ENGINE SUITE.
    
    Execution order:
    1. Single-indicator engines (geometry, entropy, wavelets, etc.)
    2. Cohort discovery from correlation substrate
    3. Cohort engines (PCA)
    4. Pairwise engines on high-correlation pairs (MI, Granger)
    5. Regime engines (HMM)
    """
    
    def __init__(self, domain: Domain = Domain.FINANCE):
        self.domain = domain
        self.conn = open_prism_db()
        self.auditor = EngineAuditor()
        self.geometry_agent = GeometrySignatureAgent()
        
        # Single-indicator adapters
        self.single_adapters: List[EngineAdapter] = [
            # CORE engines
            GeometryDescriptorAdapter(),
            BasicStatsAdapter(),
            SpectralAdapter(),
            AutocorrelationAdapter(),
            
            # CONDITIONAL engines
            EntropyAdapter(),
            WaveletAdapter(),
            HurstAdapter(),
            DMDAdapter(),
            RollingBetaAdapter(),
            TrendAdapter(),
            VolatilityAdapter(),
            StationarityAdapter(),
            
            # DIAGNOSTIC engines
            RecurrenceAdapter(),
            LyapunovAdapter(),
        ]
        
        # Pairwise adapters
        self.pairwise_adapters = [
            # CORE pairwise
            CrossCorrelationAdapter(),
            MutualInformationAdapter(),
            
            # CONDITIONAL pairwise
            GrangerAdapter(),
            CopulaAdapter(),
            CointegrationAdapter(),
            DTWAdapter(),
        ]
        
        # Cohort adapters
        self.pca_adapter = PCAAdapter()
        self.hmm_adapter = HMMAdapter()
        
        # Initialize schema
        self._init_schema()
    
    def _init_schema(self):
        """Assert required tables exist (no runtime schema creation)."""
        logger.info("Verifying schema...")
        from prism.db.schema_guard import assert_tables_exist
        assert_tables_exist(self.conn, REQUIRED_TABLES)
        logger.info("Schema verified")
    
    def get_window_schedule(self) -> List[Tuple[str, str]]:
        """Get window schedule from existing data."""
        # Try correlation matrix first
        try:
            result = self.conn.execute("""
                SELECT DISTINCT window_start, window_end 
                FROM derived.correlation_matrix
                ORDER BY window_start, window_end
            """).fetchall()
            
            if result:
                logger.info(f"Found {len(result)} windows from correlation matrix")
                return [(str(r[0]), str(r[1])) for r in result]
        except:
            pass
        
        # Try meta.engine_runs
        try:
            result = self.conn.execute("""
                SELECT DISTINCT window_start, window_end 
                FROM meta.engine_runs 
                WHERE status = 'completed'
                  AND window_start IS NOT NULL
                ORDER BY window_start
            """).fetchall()
            
            if result:
                logger.info(f"Found {len(result)} windows from engine runs")
                return [(str(r[0]), str(r[1])) for r in result]
        except:
            pass
        
        logger.warning("No existing windows found - using defaults")
        return [
            ('2018-01-01', '2018-12-31'),
            ('2019-01-01', '2019-12-31'),
            ('2020-01-01', '2020-12-31'),
            ('2021-01-01', '2021-12-31'),
            ('2022-01-01', '2022-12-31'),
            ('2023-01-01', '2023-12-31'),
            ('2024-01-01', '2024-12-31'),
        ]
    
    def get_indicators(self) -> List[str]:
        """Get list of indicator IDs from canonical data.indicators table."""
        try:
            result = self.conn.execute("""
                SELECT DISTINCT indicator_id FROM data.indicators
            """).fetchall()
            if result:
                logger.info(f"Found {len(result)} indicators from data.indicators")
                return [r[0] for r in result]
        except Exception as e:
            logger.error(f"Failed to query data.indicators: {e}")

        return []
    
    def infer_temporal_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Infer temporal profile for each indicator (native sampling support).
        
        Returns dict mapping indicator_id to:
            - native_frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'irregular'
            - median_interval_days: median days between observations
            - total_observations: total count of non-null values
            - date_range: (min_date, max_date)
        """
        profiles = {}
        
        try:
            # Get temporal stats for each indicator
            result = self.conn.execute("""
                WITH indicator_stats AS (
                    SELECT 
                        indicator_id,
                        COUNT(*) as n_obs,
                        MIN(date) as min_date,
                        MAX(date) as max_date,
                        (MAX(date) - MIN(date)) as date_span_days
                    FROM data.indicators
                    WHERE value IS NOT NULL
                    GROUP BY indicator_id
                    HAVING COUNT(*) >= 2
                )
                SELECT 
                    indicator_id,
                    n_obs,
                    min_date,
                    max_date,
                    date_span_days,
                    CASE 
                        WHEN n_obs <= 1 THEN NULL
                        ELSE CAST(date_span_days AS DOUBLE) / (n_obs - 1)
                    END as avg_interval_days
                FROM indicator_stats
            """).fetchall()
            
            for row in result:
                indicator_id, n_obs, min_date, max_date, date_span, avg_interval = row
                
                # Classify frequency based on average interval
                if avg_interval is None:
                    freq = 'irregular'
                elif avg_interval <= 1.5:
                    freq = 'daily'
                elif avg_interval <= 8:
                    freq = 'weekly'
                elif avg_interval <= 35:
                    freq = 'monthly'
                elif avg_interval <= 100:
                    freq = 'quarterly'
                else:
                    freq = 'irregular'
                
                profiles[indicator_id] = {
                    'native_frequency': freq,
                    'median_interval_days': avg_interval or 0,
                    'total_observations': n_obs,
                    'date_range': (str(min_date), str(max_date)),
                }
            
            # Log frequency distribution
            freq_counts = defaultdict(int)
            for p in profiles.values():
                freq_counts[p['native_frequency']] += 1
            logger.info(f"Temporal profiles: {dict(freq_counts)}")
            
        except Exception as e:
            logger.error(f"Failed to infer temporal profiles: {e}")
        
        return profiles
    
    def get_window_for_indicator(
        self,
        indicator_id: str,
        profile: Dict[str, Any],
        min_observations: int = 30,
        anchor_date: str = None
    ) -> Optional[Tuple[str, str]]:
        """
        Calculate appropriate window bounds for an indicator based on its frequency.
        
        For native sampling: window size adapts to ensure min_observations
        given the indicator's native frequency.
        
        Args:
            indicator_id: The indicator
            profile: Temporal profile from infer_temporal_profiles()
            min_observations: Minimum observations needed
            anchor_date: End date for window (defaults to max date in data)
            
        Returns:
            (window_start, window_end) or None if insufficient data
        """
        if profile['total_observations'] < min_observations:
            return None  # Not enough data regardless of window
        
        interval_days = profile['median_interval_days']
        if interval_days <= 0:
            interval_days = 1  # Fallback for daily
        
        # Calculate required window span
        required_days = int(interval_days * min_observations * 1.2)  # 20% buffer
        
        min_date, max_date = profile['date_range']
        
        if anchor_date is None:
            anchor_date = max_date
        
        # Calculate window start
        from datetime import datetime, timedelta
        end_dt = datetime.strptime(anchor_date, '%Y-%m-%d') if isinstance(anchor_date, str) else anchor_date
        start_dt = end_dt - timedelta(days=required_days)
        
        # Don't go before data starts
        min_dt = datetime.strptime(min_date, '%Y-%m-%d') if isinstance(min_date, str) else min_date
        if start_dt < min_dt:
            start_dt = min_dt
        
        return (start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
    
    def group_indicators_by_frequency(
        self, 
        profiles: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Group indicators by their native frequency for batch processing.
        
        Returns dict mapping frequency -> list of indicator_ids
        """
        groups = defaultdict(list)
        for indicator_id, profile in profiles.items():
            groups[profile['native_frequency']].append(indicator_id)
        
        return dict(groups)
    
    def load_series(
        self,
        indicator_id: str,
        window_start: str,
        window_end: str
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        Load native series with dates from canonical data.indicators table.
        Returns (values, dates) or (None, None).
        """
        try:
            result = self.conn.execute("""
                SELECT date, value
                FROM data.indicators
                WHERE indicator_id = ?
                  AND date >= ?
                  AND date <= ?
                ORDER BY date
            """, [indicator_id, window_start, window_end]).fetchall()

            if result:
                dates = [str(r[0]) for r in result if r[1] is not None]
                values = np.array([r[1] for r in result if r[1] is not None])
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    return values, dates[:len(values)]
        except Exception as e:
            logger.error(f"Failed to load series for {indicator_id}: {e}")

        return None, None
    
    def get_correlated_pairs(
        self, 
        window_start: str, 
        window_end: str,
        threshold: float = 0.7,
        max_pairs: int = 500
    ) -> List[Tuple[str, str]]:
        """Get highly correlated indicator pairs for pairwise analysis."""
        try:
            result = self.conn.execute("""
                SELECT indicator_id_1, indicator_id_2
                FROM derived.correlation_matrix
                WHERE window_start = ?
                  AND window_end = ?
                  AND ABS(correlation) >= ?
                  AND indicator_id_1 < indicator_id_2
                ORDER BY ABS(correlation) DESC
                LIMIT ?
            """, [window_start, window_end, threshold, max_pairs]).fetchall()
            
            return [(r[0], r[1]) for r in result]
        except:
            return []
    
    def discover_cohorts(
        self, 
        window_start: str, 
        window_end: str,
        min_cohort_size: int = 5,
        correlation_threshold: float = 0.6
    ) -> Dict[str, List[str]]:
        """
        Discover cohorts from correlation substrate.
        Simple connected-component approach.
        """
        cohorts = {}
        
        try:
            # Get adjacency
            result = self.conn.execute("""
                SELECT indicator_id_1, indicator_id_2
                FROM derived.correlation_matrix
                WHERE window_start = ?
                  AND window_end = ?
                  AND ABS(correlation) >= ?
            """, [window_start, window_end, correlation_threshold]).fetchall()
            
            if not result:
                return cohorts
            
            # Build adjacency list
            adj = defaultdict(set)
            for r in result:
                adj[r[0]].add(r[1])
                adj[r[1]].add(r[0])
            
            # Connected components
            visited = set()
            cohort_id = 0
            
            for start in adj:
                if start in visited:
                    continue
                
                # BFS
                component = []
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    component.append(node)
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                if len(component) >= min_cohort_size:
                    cohorts[f"cohort_{cohort_id}_{window_start}"] = component
                    cohort_id += 1
            
            logger.info(f"Discovered {len(cohorts)} cohorts for window {window_start}-{window_end}")
            
        except Exception as e:
            logger.warning(f"Cohort discovery failed: {e}")
        
        return cohorts
    
    def run(
        self, 
        max_indicators: Optional[int] = None, 
        max_windows: Optional[int] = None,
        skip_pairwise: bool = False,
        skip_cohort: bool = False,
        skip_regime: bool = False,
        native_sampling: bool = True
    ):
        """
        Execute the full derived phase.
        
        Args:
            max_indicators: Limit number of indicators to process
            max_windows: Limit number of windows (only used if native_sampling=False)
            skip_pairwise: Skip MI/Granger pairwise engines
            skip_cohort: Skip PCA cohort engines
            skip_regime: Skip HMM regime engines
            native_sampling: If True, use indicator-specific windows based on native frequency.
                           If False, use fixed calendar windows.
        """
        logger.info("=" * 70)
        logger.info("PRISM DERIVED PHASE - FULL ENGINE SUITE")
        logger.info(f"Mode: {'NATIVE SAMPLING' if native_sampling else 'FIXED WINDOWS'}")
        logger.info("=" * 70)
        
        indicators = self.get_indicators()
        if max_indicators:
            indicators = indicators[:max_indicators]
        
        total_stats = defaultdict(int)
        
        if native_sampling:
            # =====================================================
            # NATIVE SAMPLING MODE
            # Each indicator gets a window sized to its frequency
            # =====================================================
            logger.info("\nInferring temporal profiles...")
            profiles = self.infer_temporal_profiles()
            
            # Filter to only indicators we're processing
            profiles = {k: v for k, v in profiles.items() if k in indicators}
            
            # Group by frequency for logging
            freq_groups = self.group_indicators_by_frequency(profiles)
            for freq, inds in freq_groups.items():
                logger.info(f"  {freq}: {len(inds)} indicators")
            
            logger.info(f"\nProcessing {len(indicators)} indicators with native windows")
            logger.info(f"Single adapters: {[a.name for a in self.single_adapters]}")
            
            # =====================================================
            # PHASE 1: Single-indicator engines (native windows)
            # =====================================================
            logger.info("\n--- Phase 1: Single-Indicator Engines (Native Sampling) ---")
            
            for adapter in self.single_adapters:
                run_id = f"{adapter.name}_native_{uuid.uuid4().hex[:8]}"
                started_at = datetime.now()
                
                total_written = 0
                total_skipped = 0
                indicators_processed = 0
                min_window_date = None
                max_window_date = None

                for indicator_id in indicators:
                    profile = profiles.get(indicator_id)
                    if not profile:
                        continue
                    
                    # Get indicator-specific window
                    window_bounds = self.get_window_for_indicator(
                        indicator_id, profile, 
                        min_observations=adapter.min_observations
                    )
                    
                    if window_bounds is None:
                        log_skip(self.conn, adapter.name, indicator_id, 
                                profile['date_range'][0], profile['date_range'][1],
                                "insufficient_data_for_frequency", 
                                f"freq={profile['native_frequency']}, n_obs={profile['total_observations']}")
                        total_skipped += 1
                        continue
                    
                    window_start, window_end = window_bounds
                    # Track actual date range for logging
                    if min_window_date is None or window_start < min_window_date:
                        min_window_date = window_start
                    if max_window_date is None or window_end > max_window_date:
                        max_window_date = window_end
                    series, dates = self.load_series(indicator_id, window_start, window_end)
                    
                    if series is None or len(series) < adapter.min_observations:
                        log_skip(self.conn, adapter.name, indicator_id, window_start, window_end,
                                "insufficient_data", f"n_obs={len(series) if series is not None else 0}")
                        total_skipped += 1
                        continue
                    
                    try:
                        if isinstance(adapter, WaveletAdapter):
                            descriptors, wavelet_results = adapter.compute(series)
                            written, skipped = write_descriptors(
                                self.conn, indicator_id, window_start, window_end, descriptors, run_id)
                            w2, s2 = write_wavelet_results(
                                self.conn, indicator_id, window_start, window_end, wavelet_results, run_id)
                            written += w2
                            skipped += s2
                        else:
                            descriptors = adapter.compute(series)
                            written, skipped = write_descriptors(
                                self.conn, indicator_id, window_start, window_end, descriptors, run_id)
                        
                        total_written += written
                        total_skipped += skipped
                        indicators_processed += 1
                        
                    except Exception as e:
                        logger.debug(f"{adapter.name} failed on {indicator_id}: {e}")
                        log_skip(self.conn, adapter.name, indicator_id, window_start, window_end,
                                "computation_error", str(e))
                        total_skipped += 1
                
                # Log with actual min/max window bounds for native mode
                log_engine_run(self.conn, run_id, adapter.name,
                              min_window_date or '1900-01-01', max_window_date or '1900-01-01',
                              'completed', total_written, total_skipped, started_at,
                              {'indicators_processed': indicators_processed, 'mode': 'native_sampling'})
                
                logger.info(f"  {adapter.name}: {indicators_processed} indicators, {total_written} rows")
                total_stats[f"{adapter.name}_written"] += total_written
            
            # For cohort/pairwise in native mode, use the most common window
            # (typically the daily indicators' window which covers most data)
            if not skip_cohort or not skip_pairwise or not skip_regime:
                # Find a reasonable common window from daily indicators
                daily_indicators = freq_groups.get('daily', [])
                if daily_indicators and daily_indicators[0] in profiles:
                    ref_profile = profiles[daily_indicators[0]]
                    common_window = self.get_window_for_indicator(
                        daily_indicators[0], ref_profile, min_observations=50
                    )
                else:
                    # Fallback to most recent year
                    common_window = self.get_window_schedule()[-1] if self.get_window_schedule() else ('2024-01-01', '2024-12-31')
                
                if common_window:
                    window_start, window_end = common_window
                    logger.info(f"\nUsing common window for cohort/pairwise: {window_start} to {window_end}")
                    
                    self._run_cohort_pairwise_regime(
                        indicators, window_start, window_end, 
                        skip_pairwise, skip_cohort, skip_regime, total_stats
                    )
        
        else:
            # =====================================================
            # FIXED WINDOW MODE (original behavior)
            # =====================================================
            windows = self.get_window_schedule()
            if max_windows:
                windows = windows[:max_windows]
            
            logger.info(f"Processing {len(indicators)} indicators × {len(windows)} windows")
            logger.info(f"Single adapters: {[a.name for a in self.single_adapters]}")
            logger.info(f"Pairwise adapters: {[a.name for a in self.pairwise_adapters]}")
            
            for window_start, window_end in windows:
                logger.info(f"\n{'='*60}")
                logger.info(f"WINDOW: {window_start} to {window_end}")
                logger.info(f"{'='*60}")
                
                # =====================================================
                # PHASE 1: Single-indicator engines
                # =====================================================
                logger.info("\n--- Phase 1: Single-Indicator Engines ---")
                
                for adapter in self.single_adapters:
                    run_id = f"{adapter.name}_{window_start}_{window_end}_{uuid.uuid4().hex[:8]}"
                    started_at = datetime.now()
                    
                    window_written = 0
                    window_skipped = 0
                    indicators_processed = 0
                    
                    for indicator_id in indicators:
                        series, dates = self.load_series(indicator_id, window_start, window_end)
                        
                        if series is None or len(series) < adapter.min_observations:
                            log_skip(self.conn, adapter.name, indicator_id, window_start, window_end,
                                    "insufficient_data", f"n_obs={len(series) if series is not None else 0}")
                            continue
                        
                        try:
                            if isinstance(adapter, WaveletAdapter):
                                descriptors, wavelet_results = adapter.compute(series)
                                written, skipped = write_descriptors(
                                    self.conn, indicator_id, window_start, window_end, descriptors, run_id)
                                w2, s2 = write_wavelet_results(
                                    self.conn, indicator_id, window_start, window_end, wavelet_results, run_id)
                                written += w2
                                skipped += s2
                            else:
                                descriptors = adapter.compute(series)
                                written, skipped = write_descriptors(
                                    self.conn, indicator_id, window_start, window_end, descriptors, run_id)
                            
                            window_written += written
                            window_skipped += skipped
                            indicators_processed += 1
                            
                        except Exception as e:
                            logger.debug(f"{adapter.name} failed on {indicator_id}: {e}")
                            log_skip(self.conn, adapter.name, indicator_id, window_start, window_end,
                                    "computation_error", str(e))
                    
                    log_engine_run(self.conn, run_id, adapter.name, window_start, window_end,
                                  'completed', window_written, window_skipped, started_at,
                                  {'indicators_processed': indicators_processed})
                    
                    logger.info(f"  {adapter.name}: {indicators_processed} indicators, {window_written} rows")
                    total_stats[f"{adapter.name}_written"] += window_written
                
                # Cohort/pairwise/regime for this window
                self._run_cohort_pairwise_regime(
                    indicators, window_start, window_end,
                    skip_pairwise, skip_cohort, skip_regime, total_stats
                )
        
        # =====================================================
        # SUMMARY
        # =====================================================
        logger.info("\n" + "=" * 70)
        logger.info("DERIVED PHASE COMPLETE")
        logger.info("=" * 70)
        
        for key, value in sorted(total_stats.items()):
            logger.info(f"  {key}: {value}")
        
        self._validate_results()
    
    def _compute_correlation_matrix(
        self,
        indicators: List[str],
        window_start: str,
        window_end: str
    ) -> int:
        """
        Compute pairwise correlations for all indicators and store in derived.correlation_matrix.
        This must run before cohort/pairwise engines.
        Returns number of correlations computed.
        """
        import itertools

        run_id = f"correlation_{window_start}_{window_end}_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now()

        # Load all series for this window
        series_dict = {}
        for ind_id in indicators:
            series, _ = self.load_series(ind_id, window_start, window_end)
            if series is not None and len(series) >= 30:  # minimum for correlation
                series_dict[ind_id] = series

        if len(series_dict) < 2:
            logger.warning(f"Not enough indicators for correlation matrix: {len(series_dict)}")
            return 0

        # Compute all pairwise correlations
        valid_indicators = list(series_dict.keys())
        correlations = []

        for ind1, ind2 in itertools.combinations(valid_indicators, 2):
            s1, s2 = series_dict[ind1], series_dict[ind2]

            # Align to common length
            min_len = min(len(s1), len(s2))
            if min_len < 30:
                continue

            s1_aligned = s1[:min_len]
            s2_aligned = s2[:min_len]

            # Handle any NaN values
            mask = ~(np.isnan(s1_aligned) | np.isnan(s2_aligned))
            if mask.sum() < 30:
                continue

            try:
                corr = np.corrcoef(s1_aligned[mask], s2_aligned[mask])[0, 1]
                if not np.isnan(corr):
                    correlations.append((ind1, ind2, corr))
            except Exception:
                continue

        # Store in database
        if correlations:
            for ind1, ind2, corr in correlations:
                self.conn.execute("""
                    INSERT OR REPLACE INTO derived.correlation_matrix
                    (indicator_id_1, indicator_id_2, window_start, window_end, correlation, run_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [ind1, ind2, window_start, window_end, corr, run_id])

            log_engine_run(self.conn, run_id, "correlation_matrix", window_start, window_end,
                          'completed', len(correlations), 0, started_at,
                          {'n_indicators': len(valid_indicators), 'n_pairs': len(correlations)})

        logger.info(f"  Correlation matrix: {len(valid_indicators)} indicators, {len(correlations)} pairs")
        return len(correlations)

    def _run_cohort_pairwise_regime(
        self,
        indicators: List[str],
        window_start: str,
        window_end: str,
        skip_pairwise: bool,
        skip_cohort: bool,
        skip_regime: bool,
        total_stats: Dict[str, int]
    ):
        """Run cohort, pairwise, and regime engines for a given window."""

        # =====================================================
        # Correlation Matrix (required for cohort/pairwise)
        # =====================================================
        logger.info("\n--- Computing Correlation Matrix ---")
        n_correlations = self._compute_correlation_matrix(indicators, window_start, window_end)
        total_stats["correlation_matrix_written"] = n_correlations

        # =====================================================
        # Cohort engines (PCA)
        # =====================================================
        if not skip_cohort:
            logger.info("\n--- Cohort Engines (PCA) ---")
            
            cohorts = self.discover_cohorts(window_start, window_end)
            
            for cohort_id, cohort_indicators in cohorts.items():
                run_id = f"pca_{cohort_id}_{uuid.uuid4().hex[:8]}"
                started_at = datetime.now()
                
                # Build data matrix
                series_list = []
                valid_indicators = []
                
                for ind_id in cohort_indicators:
                    series, _ = self.load_series(ind_id, window_start, window_end)
                    if series is not None and len(series) >= self.pca_adapter.min_observations:
                        series_list.append(series)
                        valid_indicators.append(ind_id)
                
                if len(valid_indicators) < 3:
                    log_skip(self.conn, "pca", None, window_start, window_end,
                            "insufficient_cohort", f"cohort={cohort_id}, n={len(valid_indicators)}")
                    continue
                
                # Align to common length (no resampling, just truncate)
                min_len = min(len(s) for s in series_list)
                data_matrix = np.column_stack([s[:min_len] for s in series_list])
                
                try:
                    cohort_results, summary_results = self.pca_adapter.compute_cohort(
                        data_matrix, valid_indicators, cohort_id)
                    
                    written, skipped = write_cohort_descriptors(
                        self.conn, window_start, window_end, cohort_results, run_id)
                    
                    # Write summary to geometry_descriptors for the cohort
                    for desc in summary_results:
                        write_descriptors(self.conn, cohort_id, window_start, window_end, [desc], run_id)
                    
                    log_engine_run(self.conn, run_id, "pca", window_start, window_end,
                                  'completed', written, skipped, started_at,
                                  {'cohort_id': cohort_id, 'n_indicators': len(valid_indicators)})
                    
                    logger.info(f"  PCA {cohort_id}: {len(valid_indicators)} indicators, {written} rows")
                    total_stats["pca_written"] += written
                    
                except Exception as e:
                    logger.warning(f"PCA failed for {cohort_id}: {e}")
        
        # =====================================================
        # Pairwise engines (MI, Granger)
        # =====================================================
        if not skip_pairwise:
            logger.info("\n--- Pairwise Engines ---")
            
            pairs = self.get_correlated_pairs(window_start, window_end, threshold=0.6, max_pairs=200)
            
            if pairs:
                for adapter in self.pairwise_adapters:
                    run_id = f"{adapter.name}_{window_start}_{window_end}_{uuid.uuid4().hex[:8]}"
                    started_at = datetime.now()
                    
                    all_results = []
                    pairs_processed = 0
                    
                    for ind1, ind2 in pairs:
                        series1, _ = self.load_series(ind1, window_start, window_end)
                        series2, _ = self.load_series(ind2, window_start, window_end)
                        
                        if series1 is None or series2 is None:
                            continue
                        
                        if len(series1) < adapter.min_observations or len(series2) < adapter.min_observations:
                            continue
                        
                        try:
                            results = adapter.compute_pairwise(series1, series2, ind1, ind2)
                            all_results.extend(results)
                            pairs_processed += 1
                        except Exception as e:
                            logger.debug(f"{adapter.name} failed for {ind1}-{ind2}: {e}")
                    
                    written, skipped = write_pairwise_descriptors(
                        self.conn, window_start, window_end, all_results, run_id)
                    
                    log_engine_run(self.conn, run_id, adapter.name, window_start, window_end,
                                  'completed', written, skipped, started_at,
                                  {'pairs_processed': pairs_processed})
                    
                    logger.info(f"  {adapter.name}: {pairs_processed} pairs, {written} rows")
                    total_stats[f"{adapter.name}_written"] += written
        
        # =====================================================
        # Regime engines (HMM)
        # =====================================================
        if not skip_regime:
            logger.info("\n--- Regime Engines (HMM) ---")
            
            run_id = f"hmm_{window_start}_{window_end}_{uuid.uuid4().hex[:8]}"
            started_at = datetime.now()
            
            hmm_written = 0
            indicators_processed = 0
            
            # Only run HMM on first 50 indicators for performance
            for indicator_id in indicators[:50]:
                series, dates = self.load_series(indicator_id, window_start, window_end)
                
                if series is None or len(series) < self.hmm_adapter.min_observations:
                    continue
                
                try:
                    descriptors, regime_results = self.hmm_adapter.compute_regimes(
                        series, indicator_id, dates)
                    
                    written, _ = write_descriptors(
                        self.conn, indicator_id, window_start, window_end, descriptors, run_id)
                    
                    w2, _ = write_regime_results(
                        self.conn, indicator_id, window_start, window_end, regime_results, run_id)
                    
                    hmm_written += written + w2
                    indicators_processed += 1
                    
                except Exception as e:
                    logger.debug(f"HMM failed for {indicator_id}: {e}")
            
            log_engine_run(self.conn, run_id, "hmm", window_start, window_end,
                          'completed', hmm_written, 0, started_at,
                          {'indicators_processed': indicators_processed})
            
            logger.info(f"  HMM: {indicators_processed} indicators, {hmm_written} rows")
            total_stats["hmm_written"] += hmm_written
    
    def _validate_results(self):
        """Run acceptance checks."""
        logger.info("\n--- Acceptance Checks ---")
        
        checks = [
            ("geometry_descriptors", "SELECT COUNT(*) FROM derived.geometry_descriptors"),
            ("pairwise_descriptors", "SELECT COUNT(*) FROM derived.pairwise_descriptors"),
            ("cohort_descriptors", "SELECT COUNT(*) FROM derived.cohort_descriptors"),
            ("wavelet_coefficients", "SELECT COUNT(*) FROM derived.wavelet_coefficients"),
            ("regime_assignments", "SELECT COUNT(*) FROM derived.regime_assignments"),
            ("engine_runs (derived)", "SELECT COUNT(*) FROM meta.engine_runs WHERE phase = 'derived'"),
        ]
        
        for name, query in checks:
            try:
                result = self.conn.execute(query).fetchone()
                count = result[0] if result else 0
                status = "✓" if count > 0 else "○"
                logger.info(f"  {status} {name}: {count:,} rows")
            except Exception as e:
                logger.warning(f"  ✗ {name}: query failed - {e}")
        
        # Dimension summary
        try:
            result = self.conn.execute("""
                SELECT dimension, COUNT(*) as n, ROUND(AVG(value), 4) as avg_val
                FROM derived.geometry_descriptors
                GROUP BY dimension
                ORDER BY n DESC
                LIMIT 20
            """).fetchall()
            
            if result:
                logger.info("\nTop dimensions by count:")
                for row in result:
                    logger.info(f"    {row[0]}: n={row[1]:,}, avg={row[2]}")
        except:
            pass


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PRISM Derived Phase - Full Engine Suite')
    parser.add_argument('--max-indicators', type=int, default=None, help='Limit indicators')
    parser.add_argument('--max-windows', type=int, default=None, help='Limit windows (only used with --no-native-sampling)')
    parser.add_argument('--domain', type=str, default='economic',
                        choices=['economic', 'climate', 'epidemiology', 'seismology'])
    parser.add_argument('--skip-pairwise', action='store_true', help='Skip MI/Granger')
    parser.add_argument('--skip-cohort', action='store_true', help='Skip PCA')
    parser.add_argument('--skip-regime', action='store_true', help='Skip HMM')
    parser.add_argument('--full', action='store_true',
                        help='Run full derived analysis (ALL indicators, windows, engines). Omitted = minimal validation run.')
    parser.add_argument('--native-sampling', action='store_true', default=True,
                        help='Use indicator-specific windows based on native frequency (default: True)')
    parser.add_argument('--no-native-sampling', action='store_false', dest='native_sampling',
                        help='Use fixed calendar windows instead of native sampling')

    args = parser.parse_args()

    # Map 'economic' to FINANCE domain for engine gating (semantically equivalent)
    domain_map = {'economic': Domain.FINANCE, 'climate': Domain.CLIMATE, 'epidemiology': Domain.EPIDEMIOLOGY, 'seismology': Domain.SEISMOLOGY}
    domain = domain_map[args.domain]

    # Determine execution mode and apply defaults
    if args.full:
        logger.info("MODE: FULL DERIVED ANALYSIS")
        max_indicators = args.max_indicators
        max_windows = args.max_windows
        skip_pairwise = args.skip_pairwise
        skip_cohort = args.skip_cohort
        skip_regime = args.skip_regime
    else:
        logger.info("MODE: MINIMAL VALIDATION RUN (use --full for complete analysis)")
        # Minimal defaults: small scope, geometry/descriptor engines only
        max_indicators = args.max_indicators if args.max_indicators is not None else 5
        max_windows = args.max_windows if args.max_windows is not None else 1
        skip_pairwise = True
        skip_cohort = True
        skip_regime = True

    orchestrator = DerivedPhaseOrchestrator(domain=domain)

    orchestrator.run(
        max_indicators=max_indicators,
        max_windows=max_windows,
        skip_pairwise=skip_pairwise,
        skip_cohort=skip_cohort,
        skip_regime=skip_regime,
        native_sampling=args.native_sampling
    )


if __name__ == "__main__":
    main()