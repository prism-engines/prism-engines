"""
Spectral Coherence Engine
=========================

Analyzes frequency-domain relationships between indicators.
Key component for discovering shared rhythms across domains.

Coherence measures how strongly two signals are related at each frequency:
- 1.0 = Perfect linear relationship
- 0.0 = No relationship

This enables discovery of shared cycles (annual, quarterly, etc.)
regardless of the domain labels.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Try to import scipy for spectral analysis
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class CoherencePair:
    """Coherence result for a pair of indicators."""
    indicator_1: str
    indicator_2: str
    mean_coherence: float
    peak_coherence: float
    peak_frequency: float
    peak_period_days: float
    dominant_band: str
    phase_at_peak: float  # radians


class SpectralCoherenceEngine:
    """
    Spectral Coherence Analysis Engine.
    
    Discovers shared frequency relationships between indicators,
    enabling cross-domain rhythm discovery.
    """
    
    # Plugin metadata
    METADATA = {
        'name': 'Spectral Coherence Engine',
        'version': '1.0.0',
        'description': 'Frequency-domain coherence analysis',
        'author': 'PRISM Project',
        'tags': ['coherence', 'frequency', 'spectral', 'cross-domain'],
        'dependencies': ['numpy', 'scipy']
    }
    
    # Frequency bands (assuming daily data)
    FREQUENCY_BANDS = {
        'long_term': (0, 0.05),        # > 20 days
        'medium_term': (0.05, 0.15),   # 7-20 days
        'short_term': (0.15, 0.35),    # 3-7 days
        'noise': (0.35, 0.5)           # < 3 days
    }
    
    def __init__(self, 
                 nperseg: int = 256,
                 noverlap: Optional[int] = None,
                 detrend: str = 'linear'):
        """
        Initialize spectral coherence engine.
        
        Args:
            nperseg: Length of each segment for Welch's method
            noverlap: Overlap between segments (default: nperseg // 2)
            detrend: Detrending method ('linear', 'constant', False)
        """
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap else nperseg // 2
        self.detrend = detrend
        
        if not SCIPY_AVAILABLE:
            print("⚠️ scipy not available - using fallback coherence computation")
    
    def compute_coherence(self, 
                          x: np.ndarray, 
                          y: np.ndarray,
                          fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute coherence between two signals.
        
        Args:
            x, y: Input signals
            fs: Sampling frequency
            
        Returns:
            Tuple of (frequencies, coherence, phase)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Remove NaN (align)
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        n = len(x)
        if n < self.nperseg:
            # Fallback for short series
            return self._fallback_coherence(x, y, fs)
        
        if SCIPY_AVAILABLE:
            # Use scipy's coherence
            freqs, coh = signal.coherence(
                x, y, 
                fs=fs,
                nperseg=min(self.nperseg, n // 2),
                noverlap=min(self.noverlap, n // 4),
                detrend=self.detrend
            )
            
            # Compute cross-spectral density for phase
            freqs, pxy = signal.csd(
                x, y,
                fs=fs,
                nperseg=min(self.nperseg, n // 2),
                noverlap=min(self.noverlap, n // 4),
                detrend=self.detrend
            )
            
            phase = np.angle(pxy)
            
            return freqs, coh, phase
        else:
            return self._fallback_coherence(x, y, fs)
    
    def _fallback_coherence(self, 
                            x: np.ndarray, 
                            y: np.ndarray,
                            fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple coherence computation without scipy.
        Uses correlation at different lags as approximation.
        """
        n = len(x)
        
        # Standardize
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)
        y = (y - np.mean(y)) / (np.std(y) + 1e-10)
        
        # FFT-based cross-correlation
        fft_x = fft(x) if 'fft' in dir() else np.fft.fft(x)
        fft_y = fft(y) if 'fft' in dir() else np.fft.fft(y)
        
        # Cross-spectral density (simplified)
        pxy = fft_x * np.conj(fft_y)
        pxx = np.abs(fft_x) ** 2
        pyy = np.abs(fft_y) ** 2
        
        # Coherence
        coh = np.abs(pxy) ** 2 / (pxx * pyy + 1e-10)
        
        # Frequencies
        freqs = np.fft.fftfreq(n, d=1/fs)
        
        # Take positive frequencies only
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        coh = coh[pos_mask]
        phase = np.angle(pxy[pos_mask])
        
        return freqs, np.real(coh), phase
    
    def classify_frequency(self, freq: float) -> str:
        """Classify frequency into a named band."""
        for band, (low, high) in self.FREQUENCY_BANDS.items():
            if low <= freq < high:
                return band
        return 'noise'
    
    def analyze_pair(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     name_x: str,
                     name_y: str,
                     fs: float = 1.0) -> CoherencePair:
        """
        Analyze coherence between two indicators.
        
        Args:
            x, y: Time series data
            name_x, name_y: Indicator names
            fs: Sampling frequency
            
        Returns:
            CoherencePair with analysis results
        """
        freqs, coh, phase = self.compute_coherence(x, y, fs)
        
        # Mean coherence (excluding DC and noise)
        valid_mask = (freqs > 0.01) & (freqs < 0.4)
        mean_coh = np.mean(coh[valid_mask]) if np.any(valid_mask) else 0.0
        
        # Peak coherence
        if len(coh) > 0:
            peak_idx = np.argmax(coh)
            peak_coh = coh[peak_idx]
            peak_freq = freqs[peak_idx]
            peak_phase = phase[peak_idx]
        else:
            peak_coh = 0.0
            peak_freq = 0.0
            peak_phase = 0.0
        
        # Period in days (assuming daily sampling)
        peak_period = 1 / peak_freq if peak_freq > 0 else np.inf
        
        # Dominant band
        dominant_band = self.classify_frequency(peak_freq)
        
        return CoherencePair(
            indicator_1=name_x,
            indicator_2=name_y,
            mean_coherence=round(float(mean_coh), 4),
            peak_coherence=round(float(peak_coh), 4),
            peak_frequency=round(float(peak_freq), 6),
            peak_period_days=round(float(peak_period), 1),
            dominant_band=dominant_band,
            phase_at_peak=round(float(peak_phase), 4)
        )
    
    def analyze(self,
                data: pd.DataFrame,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze coherence for all indicator pairs.
        
        Args:
            data: DataFrame with indicators as columns
            
        Returns:
            Dictionary with pairwise coherence, band summary, network
        """
        columns = list(data.columns)
        n_indicators = len(columns)
        
        pairwise_results = {}
        coherence_matrix = defaultdict(dict)
        band_coherence = defaultdict(list)
        
        # Compute all pairs
        for i in range(n_indicators):
            for j in range(i + 1, n_indicators):
                col_i = columns[i]
                col_j = columns[j]
                
                x = data[col_i].values
                y = data[col_j].values
                
                result = self.analyze_pair(x, y, col_i, col_j)
                
                pair_key = f"{col_i}|{col_j}"
                pairwise_results[pair_key] = vars(result)
                
                # Symmetric matrix
                coherence_matrix[col_i][col_j] = result.mean_coherence
                coherence_matrix[col_j][col_i] = result.mean_coherence
                
                # Band analysis
                band_coherence[result.dominant_band].append({
                    'pair': (col_i, col_j),
                    'coherence': result.peak_coherence
                })
        
        # Summary by band
        band_summary = {}
        for band, pairs in band_coherence.items():
            if pairs:
                coherences = [p['coherence'] for p in pairs]
                band_summary[band] = {
                    'pair_count': len(pairs),
                    'mean_coherence': round(np.mean(coherences), 4),
                    'max_coherence': round(max(coherences), 4),
                    'top_pairs': sorted(pairs, key=lambda x: x['coherence'], reverse=True)[:3]
                }
        
        # Top pairs by band
        top_pairs_by_band = {}
        for band in self.FREQUENCY_BANDS.keys():
            if band in band_summary:
                top_pairs_by_band[band] = [
                    {'pair': p['pair'], 'coherence': p['coherence']}
                    for p in band_summary[band].get('top_pairs', [])
                ]
        
        # Network representation (for visualization)
        network = {
            'nodes': columns,
            'edges': [
                {
                    'source': r['indicator_1'],
                    'target': r['indicator_2'],
                    'weight': r['mean_coherence']
                }
                for r in pairwise_results.values()
                if r['mean_coherence'] > 0.3  # Threshold for significant edges
            ]
        }
        
        return {
            'pairwise_coherence': pairwise_results,
            'coherence_matrix': dict(coherence_matrix),
            'band_summary': band_summary,
            'top_pairs_by_band': top_pairs_by_band,
            'network': network,
            'indicators_analyzed': n_indicators,
            'pairs_analyzed': len(pairwise_results)
        }
    
    def rank_indicators(self,
                        data: pd.DataFrame,
                        by: str = 'centrality') -> List[Tuple[str, float]]:
        """
        Rank indicators by coherence characteristics.
        
        Args:
            data: DataFrame with indicators
            by: Ranking method ('centrality', 'max_coherence')
            
        Returns:
            List of (indicator, score) tuples
        """
        results = self.analyze(data)
        coherence_matrix = results['coherence_matrix']
        
        if by == 'centrality':
            # Average coherence with all other indicators
            scores = {}
            for indicator, coherences in coherence_matrix.items():
                if coherences:
                    scores[indicator] = np.mean(list(coherences.values()))
                else:
                    scores[indicator] = 0.0
        
        elif by == 'max_coherence':
            # Maximum coherence with any indicator
            scores = {}
            for indicator, coherences in coherence_matrix.items():
                if coherences:
                    scores[indicator] = max(coherences.values())
                else:
                    scores[indicator] = 0.0
        
        else:
            scores = {ind: 0.0 for ind in data.columns}
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_cycle_analysis(self,
                           series: np.ndarray,
                           fs: float = 1.0) -> Dict[str, Any]:
        """
        Analyze dominant cycles in a single series.
        
        Args:
            series: Time series data
            fs: Sampling frequency
            
        Returns:
            Dictionary with PSD and dominant cycles
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]
        
        n = len(series)
        if n < 50:
            return {'error': 'Series too short'}
        
        # Standardize
        series = (series - np.mean(series)) / (np.std(series) + 1e-10)
        
        if SCIPY_AVAILABLE:
            freqs, psd = signal.welch(
                series,
                fs=fs,
                nperseg=min(self.nperseg, n // 2),
                detrend=self.detrend
            )
        else:
            # Simple periodogram
            fft_vals = np.fft.fft(series)
            psd = np.abs(fft_vals[:n//2]) ** 2 / n
            freqs = np.fft.fftfreq(n, d=1/fs)[:n//2]
        
        # Find peaks
        valid_mask = freqs > 0.01
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]
        
        if len(valid_psd) == 0:
            return {'error': 'No valid frequencies'}
        
        # Top 3 cycles
        top_indices = np.argsort(valid_psd)[-3:][::-1]
        
        dominant_cycles = []
        for idx in top_indices:
            freq = valid_freqs[idx]
            power = valid_psd[idx]
            period = 1 / freq if freq > 0 else np.inf
            band = self.classify_frequency(freq)
            
            dominant_cycles.append({
                'frequency': round(float(freq), 6),
                'period_days': round(float(period), 1),
                'power': round(float(power), 4),
                'band': band
            })
        
        return {
            'dominant_cycles': dominant_cycles,
            'total_power': round(float(np.sum(psd)), 4)
        }


# =============================================================================
# Convenience function
# =============================================================================

def compute_coherence(x: np.ndarray, y: np.ndarray) -> float:
    """Quick coherence computation for two series."""
    engine = SpectralCoherenceEngine()
    result = engine.analyze_pair(x, y, 'x', 'y')
    return result.mean_coherence


# =============================================================================
# CLI / Standalone Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Spectral Coherence Engine - Test")
    print("=" * 60)
    
    np.random.seed(42)
    n = 500
    t = np.arange(n)
    
    # Create test signals with known relationships
    # Annual cycle (shared between climate and economic)
    annual = np.sin(2 * np.pi * t / 252)
    
    # Quarterly cycle (shared in economic)
    quarterly = np.sin(2 * np.pi * t / 63)
    
    panel = pd.DataFrame({
        # High coherence pair (both have annual cycle)
        'temp_anomaly': annual + np.random.randn(n) * 0.3,
        'energy_demand': annual * 0.8 + np.random.randn(n) * 0.4,
        
        # Medium coherence (annual + quarterly)
        'gdp_proxy': annual * 0.5 + quarterly * 0.5 + np.random.randn(n) * 0.3,
        
        # Low coherence (pure noise)
        'random_noise': np.random.randn(n),
        
        # Different cycle
        'flu_activity': np.sin(2 * np.pi * t / 365) + np.random.randn(n) * 0.2
    })
    
    print(f"\nTest panel: {panel.shape[0]} rows, {panel.shape[1]} columns")
    print(f"scipy available: {SCIPY_AVAILABLE}")
    
    # Run analysis
    engine = SpectralCoherenceEngine()
    results = engine.analyze(panel)
    
    print(f"\n📊 Pairs analyzed: {results['pairs_analyzed']}")
    
    print("\n🔗 Top Coherent Pairs:")
    pairs = results['pairwise_coherence']
    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1]['mean_coherence'], reverse=True)
    
    for pair_key, pair_data in sorted_pairs[:5]:
        print(f"   {pair_data['indicator_1']} <-> {pair_data['indicator_2']}")
        print(f"      Mean coherence: {pair_data['mean_coherence']:.3f}")
        print(f"      Peak period: {pair_data['peak_period_days']:.0f} days ({pair_data['dominant_band']})")
    
    print("\n📈 Band Summary:")
    for band, summary in results['band_summary'].items():
        print(f"   {band}: {summary['pair_count']} pairs, mean={summary['mean_coherence']:.3f}")
    
    # Verification
    print("\n" + "-" * 60)
    print("VERIFICATION:")
    
    # temp_anomaly and energy_demand should be highly coherent
    pair_key = "temp_anomaly|energy_demand"
    if pair_key in pairs:
        coh = pairs[pair_key]['mean_coherence']
        if coh > 0.5:
            print(f"✅ temp_anomaly <-> energy_demand: coherence={coh:.3f} (expected high)")
        else:
            print(f"⚠️ temp_anomaly <-> energy_demand: coherence={coh:.3f} (expected > 0.5)")
    
    # random_noise should have low coherence with everything
    noise_pairs = [p for k, p in pairs.items() if 'random_noise' in k]
    if noise_pairs:
        max_noise_coh = max(p['mean_coherence'] for p in noise_pairs)
        if max_noise_coh < 0.3:
            print(f"✅ random_noise max coherence: {max_noise_coh:.3f} (expected low)")
        else:
            print(f"⚠️ random_noise max coherence: {max_noise_coh:.3f} (expected < 0.3)")
    
    print("\n" + "=" * 60)
    print("Test completed!")
