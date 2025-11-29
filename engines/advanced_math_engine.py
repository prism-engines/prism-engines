# VCF Research: Advanced Mathematical Models
# Vector Coherence & Harmonic Variance Framework

"""
This module implements sophisticated mathematical models for:
1. Vector variance decomposition
2. Harmonic coherence analysis
3. Dynamic mode analysis
4. Multi-scale synchronization
5. Geometric manifold dynamics
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import hilbert, stft, istft
from scipy.linalg import svd, eig
from sklearn.decomposition import PCA
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: VECTOR VARIANCE MODELS
# ============================================================================

class VectorVarianceDecomposition:
    """
    Decompose variance into geometric components:
    - Directional variance (rotational)
    - Magnitude variance (radial)
    - Angular momentum
    - Vector field divergence/curl
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        """
        panel_df: DataFrame where each column is a time series
        """
        self.panel = panel_df
        self.n_series = panel_df.shape[1]
        self.n_time = panel_df.shape[0]
        
    def compute_vector_field(self) -> np.ndarray:
        """
        Treat each time point as a vector in N-dimensional space
        Returns: (n_time, n_series) array
        """
        return self.panel.values
    
    def magnitude_variance(self) -> pd.Series:
        """
        Variance in vector magnitude over time
        ||v(t)||² variance
        """
        vectors = self.compute_vector_field()
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        return pd.Series({
            'mean_magnitude': np.mean(magnitudes),
            'std_magnitude': np.std(magnitudes),
            'variance_magnitude': np.var(magnitudes),
            'cv_magnitude': np.std(magnitudes) / np.mean(magnitudes)  # Coefficient of variation
        })
    
    def directional_variance(self) -> Dict:
        """
        Measure how much the direction changes (rotational variance)
        Uses angular differences between consecutive vectors
        """
        vectors = self.compute_vector_field()
        
        # Normalize to unit vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / (norms + 1e-10)
        
        # Angular differences
        angular_diffs = []
        for i in range(len(unit_vectors) - 1):
            cos_angle = np.dot(unit_vectors[i], unit_vectors[i+1])
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angular_diffs.append(angle)
        
        angular_diffs = np.array(angular_diffs)
        
        return {
            'mean_angular_change': np.mean(angular_diffs),
            'std_angular_change': np.std(angular_diffs),
            'total_rotation': np.sum(angular_diffs),
            'rotation_rate': np.mean(angular_diffs)
        }
    
    def angular_momentum(self, window: int = 12) -> pd.Series:
        """
        Measure rotational momentum in vector space
        L = r × v (cross product analog in high dimensions)
        """
        vectors = self.compute_vector_field()
        velocities = np.diff(vectors, axis=0)
        
        # Rolling angular momentum
        momentum = []
        for i in range(len(velocities)):
            if i < window:
                continue
            
            # Local rotation measure
            local_vectors = vectors[i-window:i+1]
            rotation = np.std([np.arctan2(v[1], v[0]) if len(v) >= 2 else 0 
                              for v in local_vectors])
            momentum.append(rotation)
        
        return pd.Series(momentum, index=self.panel.index[window:])
    
    def vector_divergence(self, window: int = 12) -> pd.Series:
        """
        Measure if vectors are expanding (divergence > 0) or contracting (< 0)
        Similar to div(F) in vector calculus
        """
        vectors = self.compute_vector_field()
        
        divergence = []
        for i in range(window, len(vectors)):
            local = vectors[i-window:i+1]
            
            # Measure expansion/contraction
            center = np.mean(local, axis=0)
            distances = [np.linalg.norm(v - center) for v in local]
            
            # Positive divergence = expanding
            div = (distances[-1] - distances[0]) / window
            divergence.append(div)
        
        return pd.Series(divergence, index=self.panel.index[window:])
    
    def vector_curl(self, col_a: str, col_b: str) -> pd.Series:
        """
        2D curl analog: rotation in the plane defined by two series
        Measures circular motion in phase space
        """
        x = self.panel[col_a].values
        y = self.panel[col_b].values
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # 2D curl = ∂y/∂x - ∂x/∂y (approximated)
        curl = dy[1:] - dx[1:]
        
        return pd.Series(curl, index=self.panel.index[1:])
    
    def explained_variance_by_dimension(self) -> pd.DataFrame:
        """
        PCA-style variance decomposition
        Shows which 'eigendirections' explain most variance
        """
        vectors = self.compute_vector_field()
        
        # Center the data
        centered = vectors - np.mean(vectors, axis=0)
        
        # Covariance matrix
        cov = np.cov(centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eig(cov)
        eigenvalues = np.real(eigenvalues)
        
        # Sort by explained variance
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        
        total_var = np.sum(eigenvalues)
        explained_var = eigenvalues / total_var
        
        return pd.DataFrame({
            'eigenvalue': eigenvalues,
            'explained_variance': explained_var,
            'cumulative_variance': np.cumsum(explained_var)
        })


# ============================================================================
# PART 2: HARMONIC COHERENCE MODELS
# ============================================================================

class HarmonicCoherence:
    """
    Advanced harmonic analysis for coherence measurement
    - Wavelet coherence (time-frequency)
    - Cross-spectral density
    - Phase-locking value
    - Frequency-specific synchronization
    """
    
    @staticmethod
    def wavelet_coherence(signal_a: np.ndarray, signal_b: np.ndarray, 
                         scales: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time-frequency coherence using continuous wavelet transform
        Shows which frequencies are coherent at which times
        """
        if scales is None:
            scales = np.arange(1, min(128, len(signal_a)//4))
        
        import pywt
        
        # Continuous wavelet transform
        coef_a, freqs_a = pywt.cwt(signal_a, scales, 'morl')
        coef_b, freqs_b = pywt.cwt(signal_b, scales, 'morl')
        
        # Cross-wavelet spectrum
        cross_spectrum = coef_a * np.conj(coef_b)
        
        # Wavelet coherence
        coherence = np.abs(cross_spectrum)**2 / (np.abs(coef_a)**2 * np.abs(coef_b)**2 + 1e-10)
        
        return coherence, scales
    
    @staticmethod
    def phase_locking_value(phase_a: np.ndarray, phase_b: np.ndarray, 
                           window: int = 50) -> pd.Series:
        """
        PLV: Measures consistency of phase relationship over time
        1 = perfect phase locking, 0 = random
        """
        phase_diff = phase_b - phase_a
        
        plv_series = []
        for i in range(window, len(phase_diff)):
            local_diff = phase_diff[i-window:i]
            plv = np.abs(np.mean(np.exp(1j * local_diff)))
            plv_series.append(plv)
        
        return pd.Series(plv_series)
    
    @staticmethod
    def cross_spectral_density(signal_a: np.ndarray, signal_b: np.ndarray, 
                               fs: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frequency-domain correlation
        Shows which frequencies are most coherent
        """
        from scipy.signal import csd
        
        f, Pxy = csd(signal_a, signal_b, fs=fs, nperseg=min(256, len(signal_a)//2))
        
        return f, Pxy
    
    @staticmethod
    def instantaneous_coherence(signal_a: np.ndarray, signal_b: np.ndarray, 
                                window: int = 50) -> pd.Series:
        """
        Rolling coherence measure
        Detects when signals move in/out of sync
        """
        from scipy.signal import coherence
        
        coh_series = []
        for i in range(window, len(signal_a)):
            local_a = signal_a[i-window:i]
            local_b = signal_b[i-window:i]
            
            f, Cxy = coherence(local_a, local_b, fs=1.0, nperseg=min(32, window//2))
            coh_series.append(np.mean(Cxy))
        
        return pd.Series(coh_series)
    
    @staticmethod
    def frequency_band_coherence(signal_a: np.ndarray, signal_b: np.ndarray,
                                 bands: Dict[str, Tuple[float, float]] = None) -> Dict:
        """
        Coherence in specific frequency bands
        e.g., 'business_cycle': (1/96, 1/18) months
        """
        from scipy.signal import butter, filtfilt, coherence
        
        if bands is None:
            # Default economic frequency bands
            bands = {
                'high_freq': (0.1, 0.5),      # < 10 periods
                'business_cycle': (1/96, 1/18),  # 1.5-8 years
                'low_freq': (0, 1/96)          # > 8 years
            }
        
        results = {}
        
        for band_name, (low, high) in bands.items():
            # Bandpass filter
            if low == 0:
                b, a = butter(2, high, btype='low', fs=1.0)
            else:
                b, a = butter(2, [low, high], btype='band', fs=1.0)
            
            filtered_a = filtfilt(b, a, signal_a)
            filtered_b = filtfilt(b, a, signal_b)
            
            # Coherence in this band
            f, Cxy = coherence(filtered_a, filtered_b, fs=1.0)
            
            results[band_name] = {
                'mean_coherence': np.mean(Cxy),
                'max_coherence': np.max(Cxy),
                'coherence_std': np.std(Cxy)
            }
        
        return results


# ============================================================================
# PART 3: DYNAMIC MODE DECOMPOSITION (DMD)
# ============================================================================

class DynamicModeDecomposition:
    """
    DMD extracts spatiotemporal coherent structures
    Discovers underlying dynamics from data
    
    Perfect for finding:
    - Oscillatory modes in macro data
    - Growth/decay rates
    - Dominant frequencies
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        self.panel = panel_df
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None
        
    def compute_dmd(self, rank: int = None) -> Dict:
        """
        Standard DMD algorithm
        X' = A X  (find A via DMD)
        """
        X = self.panel.values.T  # (n_series, n_time)
        
        # Split into snapshots
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        
        # SVD of X1
        U, s, Vt = svd(X1, full_matrices=False)
        
        if rank:
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
        
        # DMD operator
        S_inv = np.diag(1.0 / s)
        A_tilde = U.T @ X2 @ Vt.T @ S_inv
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eig(A_tilde)
        
        # DMD modes
        modes = X2 @ Vt.T @ S_inv @ eigenvectors
        
        # Amplitudes
        amplitudes = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]
        
        self.modes = modes
        self.eigenvalues = eigenvalues
        self.amplitudes = amplitudes
        
        return {
            'modes': modes,
            'eigenvalues': eigenvalues,
            'amplitudes': amplitudes,
            'frequencies': np.log(eigenvalues).imag / (2 * np.pi),
            'growth_rates': np.log(np.abs(eigenvalues))
        }
    
    def reconstruct(self, mode_indices: List[int] = None) -> pd.DataFrame:
        """
        Reconstruct time series using selected modes
        """
        if self.modes is None:
            raise ValueError("Must run compute_dmd() first")
        
        if mode_indices is None:
            mode_indices = range(len(self.eigenvalues))
        
        n_time = self.panel.shape[0]
        time_dynamics = np.array([
            self.amplitudes[i] * (self.eigenvalues[i] ** np.arange(n_time))
            for i in mode_indices
        ])
        
        X_reconstructed = (self.modes[:, mode_indices] @ time_dynamics).real
        
        return pd.DataFrame(
            X_reconstructed.T,
            columns=self.panel.columns,
            index=self.panel.index
        )
    
    def dominant_modes(self, n_modes: int = 5) -> pd.DataFrame:
        """
        Find modes with largest amplitudes (most important dynamics)
        """
        if self.modes is None:
            raise ValueError("Must run compute_dmd() first")
        
        mode_power = np.abs(self.amplitudes)
        top_indices = np.argsort(mode_power)[-n_modes:][::-1]
        
        results = []
        for idx in top_indices:
            freq = np.log(self.eigenvalues[idx]).imag / (2 * np.pi)
            growth = np.log(np.abs(self.eigenvalues[idx]))
            
            results.append({
                'mode_index': idx,
                'amplitude': np.abs(self.amplitudes[idx]),
                'frequency': freq,
                'period': 1.0 / freq if freq != 0 else np.inf,
                'growth_rate': growth,
                'stable': growth < 0
            })
        
        return pd.DataFrame(results)


# ============================================================================
# PART 4: MULTI-SCALE COHERENCE
# ============================================================================

class MultiScaleCoherence:
    """
    Analyze coherence across different time scales
    Uses empirical mode decomposition (EMD)
    """
    
    @staticmethod
    def empirical_mode_decomposition(signal: np.ndarray, max_imf: int = 5) -> List[np.ndarray]:
        """
        Decompose signal into intrinsic mode functions (IMFs)
        Each IMF represents a different time scale
        """
        from scipy.signal import hilbert
        
        imfs = []
        residual = signal.copy()
        
        for _ in range(max_imf):
            # Simple EMD implementation (sifting process)
            imf = MultiScaleCoherence._sift(residual)
            
            if imf is None:
                break
                
            imfs.append(imf)
            residual = residual - imf
            
            if np.std(residual) < 0.01 * np.std(signal):
                break
        
        imfs.append(residual)  # Trend
        return imfs
    
    @staticmethod
    def _sift(signal: np.ndarray, max_iter: int = 10) -> np.ndarray:
        """
        EMD sifting iteration
        """
        from scipy.interpolate import CubicSpline
        
        h = signal.copy()
        
        for _ in range(max_iter):
            # Find extrema
            peaks = []
            troughs = []
            
            for i in range(1, len(h) - 1):
                if h[i] > h[i-1] and h[i] > h[i+1]:
                    peaks.append(i)
                elif h[i] < h[i-1] and h[i] < h[i+1]:
                    troughs.append(i)
            
            if len(peaks) < 3 or len(troughs) < 3:
                return None
            
            # Cubic spline envelopes
            upper_env = CubicSpline(peaks, h[peaks], extrapolate=True)
            lower_env = CubicSpline(troughs, h[troughs], extrapolate=True)
            
            # Mean envelope
            x = np.arange(len(h))
            mean_env = (upper_env(x) + lower_env(x)) / 2
            
            h_new = h - mean_env
            
            # Check stopping criterion
            if np.sum((h - h_new)**2) / np.sum(h**2) < 0.01:
                return h_new
            
            h = h_new
        
        return h
    
    @staticmethod
    def scale_coherence(signal_a: np.ndarray, signal_b: np.ndarray, 
                       max_scales: int = 5) -> pd.DataFrame:
        """
        Compute coherence at each time scale (IMF level)
        """
        from scipy.signal import coherence
        
        imfs_a = MultiScaleCoherence.empirical_mode_decomposition(signal_a, max_scales)
        imfs_b = MultiScaleCoherence.empirical_mode_decomposition(signal_b, max_scales)
        
        results = []
        
        for i, (imf_a, imf_b) in enumerate(zip(imfs_a, imfs_b)):
            # Ensure same length
            min_len = min(len(imf_a), len(imf_b))
            imf_a = imf_a[:min_len]
            imf_b = imf_b[:min_len]
            
            # Coherence for this scale
            f, Cxy = coherence(imf_a, imf_b, fs=1.0)
            
            # Typical period for this IMF
            zero_crossings = np.where(np.diff(np.sign(imf_a)))[0]
            period = 2 * np.mean(np.diff(zero_crossings)) if len(zero_crossings) > 1 else np.nan
            
            results.append({
                'scale': i + 1,
                'coherence': np.mean(Cxy),
                'max_coherence': np.max(Cxy),
                'typical_period': period
            })
        
        return pd.DataFrame(results)


# ============================================================================
# PART 5: INTEGRATED ANALYSIS CLASS
# ============================================================================

class VCFMathEngine:
    """
    Unified interface for all VCF mathematical models
    """
    
    def __init__(self, panel_df: pd.DataFrame):
        self.panel = panel_df
        self.vector_variance = VectorVarianceDecomposition(panel_df)
        self.dmd = DynamicModeDecomposition(panel_df)
        
    def full_vector_analysis(self) -> Dict:
        """
        Complete vector variance decomposition
        """
        return {
            'magnitude_variance': self.vector_variance.magnitude_variance(),
            'directional_variance': self.vector_variance.directional_variance(),
            'angular_momentum': self.vector_variance.angular_momentum(),
            'divergence': self.vector_variance.vector_divergence(),
            'explained_variance': self.vector_variance.explained_variance_by_dimension()
        }
    
    def full_harmonic_analysis(self, col_a: str, col_b: str) -> Dict:
        """
        Complete harmonic coherence analysis for two series
        """
        signal_a = self.panel[col_a].values
        signal_b = self.panel[col_b].values
        
        # Phase angles
        phase_a = np.angle(hilbert(signal_a))
        phase_b = np.angle(hilbert(signal_b))
        
        return {
            'phase_locking': HarmonicCoherence.phase_locking_value(phase_a, phase_b),
            'cross_spectrum': HarmonicCoherence.cross_spectral_density(signal_a, signal_b),
            'instantaneous_coh': HarmonicCoherence.instantaneous_coherence(signal_a, signal_b),
            'frequency_bands': HarmonicCoherence.frequency_band_coherence(signal_a, signal_b),
            'multi_scale': MultiScaleCoherence.scale_coherence(signal_a, signal_b)
        }
    
    def dynamic_modes_analysis(self, n_modes: int = 5) -> Dict:
        """
        DMD analysis to find dominant temporal patterns
        """
        dmd_results = self.dmd.compute_dmd()
        dominant = self.dmd.dominant_modes(n_modes)
        
        return {
            'dmd_results': dmd_results,
            'dominant_modes': dominant,
            'reconstruction': self.dmd.reconstruct(list(range(n_modes)))
        }
    
    def regime_detection(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Detect regime changes based on coherence breaks
        """
        # Compute pairwise coherence over time
        cols = self.panel.columns
        regime_signals = []
        
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                coh = HarmonicCoherence.instantaneous_coherence(
                    self.panel[cols[i]].values,
                    self.panel[cols[j]].values,
                    window=50
                )
                regime_signals.append(coh)
        
        # Average coherence across all pairs
        avg_coherence = np.mean(regime_signals, axis=0)
        
        # Regime changes = sharp drops in coherence
        regime_changes = np.abs(np.diff(avg_coherence)) > threshold
        
        return pd.DataFrame({
            'coherence': avg_coherence,
            'regime_change': np.concatenate([[False], regime_changes])
        }, index=self.panel.index[50:])


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of all models
    """
    
    # Create sample panel (replace with your data)
    dates = pd.date_range('2010-01-01', periods=200, freq='M')
    panel = pd.DataFrame({
        'CPI': np.cumsum(np.random.randn(200) * 0.2) + 100,
        'Yield_10Y': 3 + np.sin(np.linspace(0, 8*np.pi, 200)) + np.random.randn(200) * 0.3,
        'XLF': np.cumsum(np.random.randn(200) * 2) + 50,
        'XLE': np.cumsum(np.random.randn(200) * 2.5) + 45
    }, index=dates)
    
    # Initialize engine
    engine = VCFMathEngine(panel)
    
    print("=" * 70)
    print("VCF MATHEMATICAL ANALYSIS")
    print("=" * 70)
    
    # 1. Vector Analysis
    print("\n1. VECTOR VARIANCE DECOMPOSITION")
    print("-" * 70)
    vector_results = engine.full_vector_analysis()
    print("\nMagnitude Variance:")
    print(vector_results['magnitude_variance'])
    print("\nDirectional Variance:")
    print(vector_results['directional_variance'])
    
    # 2. Harmonic Analysis
    print("\n2. HARMONIC COHERENCE ANALYSIS (CPI vs Yield)")
    print("-" * 70)
    harmonic_results = engine.full_harmonic_analysis('CPI', 'Yield_10Y')
    print("\nFrequency Band Coherence:")
    print(harmonic_results['frequency_bands'])
    
    # 3. DMD Analysis
    print("\n3. DYNAMIC MODE DECOMPOSITION")
    print("-" * 70)
    dmd_results = engine.dynamic_modes_analysis(n_modes=3)
    print("\nDominant Modes:")
    print(dmd_results['dominant_modes'])
    
    # 4. Regime Detection
    print("\n4. REGIME DETECTION")
    print("-" * 70)
    regimes = engine.regime_detection(threshold=0.3)
    n_regimes = regimes['regime_change'].sum()
    print(f"\nDetected {n_regimes} regime changes")
    print(f"Mean coherence: {regimes['coherence'].mean():.3f}")
    
    print("\n" + "=" * 70)
    print("Analysis complete. All models ready for production use.")
    print("=" * 70)
