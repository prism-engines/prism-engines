"""
Wavelet Analysis Lens
=====================
Reveals time-varying periodicities and multi-scale patterns by decomposing
time series into frequency components at different time scales.

Key Applications:
- ENSO cycle detection in climate data
- Market regime identification across multiple timeframes
- Transient event detection (crashes, spikes)

Methods:
- Discrete Wavelet Transform (DWT): Multi-resolution decomposition
- Continuous Wavelet Transform (CWT): Time-frequency analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class WaveletResult:
    """Container for wavelet analysis results."""
    coefficients: np.ndarray
    scales: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray
    time_index: np.ndarray
    metadata: Dict


class WaveletLens:
    """
    Wavelet Analysis Lens for multi-scale time series decomposition.
    
    This lens reveals patterns that exist at different time scales simultaneously,
    making it ideal for detecting:
    - Cyclic behavior (ENSO, business cycles, seasonal patterns)
    - Localized events (market shocks, extreme weather)
    - Scale-dependent correlations between variables
    """
    
    def __init__(self, wavelet: str = 'morlet', sampling_rate: float = 1.0):
        """
        Initialize the Wavelet Lens.
        
        Parameters
        ----------
        wavelet : str
            Wavelet type: 'morlet', 'mexican_hat', 'paul', 'dog' (derivative of gaussian)
        sampling_rate : float
            Sampling rate of the data (samples per unit time)
        """
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate
        self._wavelet_params = self._get_wavelet_params(wavelet)
    
    def _get_wavelet_params(self, wavelet: str) -> Dict:
        """Get wavelet-specific parameters."""
        params = {
            'morlet': {'omega0': 6.0, 'fourier_factor': 4 * np.pi / (6.0 + np.sqrt(2 + 6.0**2))},
            'mexican_hat': {'fourier_factor': 2 * np.pi / np.sqrt(2.5)},
            'paul': {'m': 4, 'fourier_factor': 4 * np.pi / (2 * 4 + 1)},
            'dog': {'m': 2, 'fourier_factor': 2 * np.pi / np.sqrt(2 + 0.5)}
        }
        if wavelet not in params:
            raise ValueError(f"Unknown wavelet: {wavelet}. Choose from {list(params.keys())}")
        return params[wavelet]
    
    def _morlet_wavelet(self, eta: np.ndarray, omega0: float = 6.0) -> np.ndarray:
        """
        Morlet wavelet function.
        
        The Morlet wavelet is a complex exponential modulated by a Gaussian,
        excellent for detecting oscillatory patterns.
        """
        return np.pi**(-0.25) * np.exp(1j * omega0 * eta) * np.exp(-eta**2 / 2)
    
    def _mexican_hat_wavelet(self, eta: np.ndarray) -> np.ndarray:
        """
        Mexican Hat (Ricker) wavelet function.
        
        Second derivative of Gaussian, good for detecting peaks and edges.
        """
        return (2 / np.sqrt(3) * np.pi**(-0.25)) * (1 - eta**2) * np.exp(-eta**2 / 2)
    
    def _paul_wavelet(self, eta: np.ndarray, m: int = 4) -> np.ndarray:
        """
        Paul wavelet function.
        
        Good for sharp signal features, asymmetric.
        """
        norm = 2**m * np.math.factorial(m) / np.sqrt(np.pi * np.math.factorial(2*m))
        return norm * (1 - 1j * eta)**(-m - 1)
    
    def _dog_wavelet(self, eta: np.ndarray, m: int = 2) -> np.ndarray:
        """
        Derivative of Gaussian (DOG) wavelet.
        
        m=2 gives Mexican Hat, higher m gives more oscillations.
        """
        if m == 2:
            return self._mexican_hat_wavelet(eta)
        norm = (-1)**(m+1) / np.sqrt(np.math.gamma(m + 0.5))
        # Compute m-th derivative of Gaussian numerically
        return norm * np.exp(-eta**2 / 2) * np.polynomial.hermite.hermval(eta, [0]*m + [1])
    
    def continuous_wavelet_transform(
        self,
        data: np.ndarray,
        scales: Optional[np.ndarray] = None,
        num_scales: int = 64
    ) -> WaveletResult:
        """
        Compute the Continuous Wavelet Transform (CWT).
        
        The CWT provides a time-frequency representation showing how
        different frequency components evolve over time.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series (1D array)
        scales : np.ndarray, optional
            Wavelet scales to use. If None, computed automatically.
        num_scales : int
            Number of scales if auto-computing
            
        Returns
        -------
        WaveletResult
            Contains coefficients, scales, frequencies, and power spectrum
        """
        n = len(data)
        
        # Remove mean and normalize
        data_norm = (data - np.mean(data)) / np.std(data)
        
        # Compute scales if not provided (logarithmically spaced)
        if scales is None:
            s0 = 2 / self.sampling_rate  # Smallest scale
            dj = 0.25  # Scale spacing
            j1 = int(np.log2(n * self._wavelet_params['fourier_factor'] / s0) / dj)
            scales = s0 * 2**(dj * np.arange(0, min(j1, num_scales)))
        
        # Pad data to power of 2 for FFT efficiency
        npad = int(2**np.ceil(np.log2(n)))
        data_padded = np.zeros(npad)
        data_padded[:n] = data_norm
        
        # Compute FFT of padded data
        data_fft = np.fft.fft(data_padded)
        freqs = np.fft.fftfreq(npad, 1/self.sampling_rate)
        
        # Initialize output array
        coefficients = np.zeros((len(scales), n), dtype=complex)
        
        # Compute CWT via convolution in frequency domain
        for i, scale in enumerate(scales):
            # Construct wavelet in frequency domain
            omega = 2 * np.pi * freqs * scale
            
            if self.wavelet == 'morlet':
                omega0 = self._wavelet_params['omega0']
                wavelet_fft = np.pi**(-0.25) * np.exp(-(omega - omega0)**2 / 2)
                wavelet_fft[freqs < 0] = 0  # Analytic wavelet
            elif self.wavelet == 'mexican_hat':
                wavelet_fft = np.sqrt(8/3) * np.pi**0.25 * omega**2 * np.exp(-omega**2 / 2)
            elif self.wavelet == 'paul':
                m = self._wavelet_params['m']
                wavelet_fft = np.where(omega > 0, 
                    2**m / np.sqrt(m * np.math.factorial(2*m-1)) * omega**m * np.exp(-omega), 0)
            else:
                raise NotImplementedError(f"CWT not implemented for {self.wavelet}")
            
            # Normalize wavelet
            wavelet_fft *= np.sqrt(2 * np.pi * scale / (1/self.sampling_rate))
            
            # Convolution via multiplication in frequency domain
            conv = np.fft.ifft(data_fft * wavelet_fft)
            coefficients[i, :] = conv[:n]
        
        # Compute power and convert scales to frequencies
        power = np.abs(coefficients)**2
        frequencies = 1 / (scales * self._wavelet_params['fourier_factor'])
        
        return WaveletResult(
            coefficients=coefficients,
            scales=scales,
            frequencies=frequencies,
            power=power,
            time_index=np.arange(n) / self.sampling_rate,
            metadata={'wavelet': self.wavelet, 'transform': 'CWT'}
        )
    
    def discrete_wavelet_transform(
        self,
        data: np.ndarray,
        levels: Optional[int] = None,
        mode: str = 'symmetric'
    ) -> Dict[str, np.ndarray]:
        """
        Compute the Discrete Wavelet Transform (DWT) using Haar wavelet.
        
        The DWT provides an efficient multi-resolution decomposition,
        separating the signal into approximation (low-freq) and detail
        (high-freq) components at each level.
        
        Parameters
        ----------
        data : np.ndarray
            Input time series
        levels : int, optional
            Number of decomposition levels. If None, max possible.
        mode : str
            Signal extension mode: 'symmetric', 'periodic', 'zero'
            
        Returns
        -------
        dict
            'approximation': final low-frequency component
            'details': list of detail coefficients at each level
            'reconstruction': reconstructed signal per level
        """
        n = len(data)
        
        # Determine max levels
        max_levels = int(np.log2(n))
        if levels is None:
            levels = min(max_levels, 6)
        levels = min(levels, max_levels)
        
        # Haar wavelet filters
        h_low = np.array([1, 1]) / np.sqrt(2)   # Lowpass (scaling)
        h_high = np.array([1, -1]) / np.sqrt(2)  # Highpass (wavelet)
        
        def extend_signal(sig, length, mode):
            """Extend signal for convolution."""
            if mode == 'symmetric':
                return np.concatenate([sig[:length][::-1], sig, sig[-length:][::-1]])
            elif mode == 'periodic':
                return np.tile(sig, 3)[n-length:2*n+length]
            else:  # zero
                return np.concatenate([np.zeros(length), sig, np.zeros(length)])
        
        def dwt_step(signal):
            """Single level DWT decomposition."""
            n_sig = len(signal)
            extended = extend_signal(signal, 1, mode)
            
            # Convolve and downsample
            approx = np.convolve(extended, h_low, 'same')[1:n_sig+1:2]
            detail = np.convolve(extended, h_high, 'same')[1:n_sig+1:2]
            
            return approx, detail
        
        # Perform multi-level decomposition
        approximation = data.copy()
        details = []
        
        for level in range(levels):
            approximation, detail = dwt_step(approximation)
            details.append(detail)
        
        # Compute reconstruction at each level for analysis
        reconstructions = self._reconstruct_levels(approximation, details, n)
        
        return {
            'approximation': approximation,
            'details': details,
            'reconstructions': reconstructions,
            'levels': levels,
            'metadata': {'wavelet': 'haar', 'transform': 'DWT', 'mode': mode}
        }
    
    def _reconstruct_levels(
        self,
        approximation: np.ndarray,
        details: List[np.ndarray],
        original_length: int
    ) -> Dict[str, np.ndarray]:
        """Reconstruct signal components at each resolution level."""
        # Haar reconstruction filters
        g_low = np.array([1, 1]) / np.sqrt(2)
        g_high = np.array([1, -1]) / np.sqrt(2)
        
        def idwt_step(approx, detail):
            """Single level inverse DWT."""
            n_out = 2 * len(approx)
            
            # Upsample
            approx_up = np.zeros(n_out)
            detail_up = np.zeros(n_out)
            approx_up[::2] = approx
            detail_up[::2] = detail
            
            # Convolve with reconstruction filters
            reconstructed = np.convolve(approx_up, g_low, 'same') + \
                           np.convolve(detail_up, g_high, 'same')
            
            return reconstructed
        
        # Reconstruct each level
        levels = len(details)
        reconstructions = {}
        
        # Start from coarsest level
        current_approx = approximation
        for level in range(levels - 1, -1, -1):
            zero_detail = np.zeros_like(details[level])
            
            # Reconstruct approximation only (trend at this scale)
            approx_recon = current_approx
            for l in range(level, -1, -1):
                approx_recon = idwt_step(approx_recon, np.zeros(len(approx_recon)))
            reconstructions[f'trend_level_{level}'] = approx_recon[:original_length]
            
            # Reconstruct with this level's detail
            current_approx = idwt_step(current_approx, details[level])
        
        return reconstructions
    
    def compute_wavelet_coherence(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        scales: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute wavelet coherence between two time series.
        
        Wavelet coherence reveals time-frequency localized correlations,
        showing when and at what frequencies two signals are correlated.
        
        Parameters
        ----------
        data1, data2 : np.ndarray
            Input time series (must be same length)
        scales : np.ndarray, optional
            Wavelet scales
            
        Returns
        -------
        dict
            'coherence': coherence values (0-1)
            'phase': phase relationship
            'scales': wavelet scales
            'frequencies': corresponding frequencies
        """
        if len(data1) != len(data2):
            raise ValueError("Time series must have equal length")
        
        # Compute CWT for both series
        cwt1 = self.continuous_wavelet_transform(data1, scales)
        cwt2 = self.continuous_wavelet_transform(data2, scales)
        
        # Cross-wavelet spectrum
        W12 = cwt1.coefficients * np.conj(cwt2.coefficients)
        
        # Smooth spectra (using moving average as simple smoothing)
        def smooth(x, window=5):
            kernel = np.ones(window) / window
            return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 1, x)
        
        S11 = smooth(np.abs(cwt1.coefficients)**2)
        S22 = smooth(np.abs(cwt2.coefficients)**2)
        S12 = smooth(W12)
        
        # Wavelet coherence
        coherence = np.abs(S12)**2 / (S11 * S22 + 1e-10)
        
        # Phase relationship
        phase = np.angle(S12)
        
        return {
            'coherence': coherence,
            'phase': phase,
            'scales': cwt1.scales,
            'frequencies': cwt1.frequencies,
            'time_index': cwt1.time_index
        }
    
    def detect_scale_patterns(
        self,
        cwt_result: WaveletResult,
        threshold: float = 0.8
    ) -> Dict[str, List]:
        """
        Detect dominant patterns at different scales.
        
        Parameters
        ----------
        cwt_result : WaveletResult
            Output from continuous_wavelet_transform
        threshold : float
            Power threshold (relative to max) for pattern detection
            
        Returns
        -------
        dict
            'dominant_scales': scales with significant power
            'dominant_frequencies': corresponding frequencies
            'peak_times': times of peak power at each scale
        """
        power = cwt_result.power
        max_power = np.max(power)
        
        dominant_scales = []
        dominant_frequencies = []
        peak_times = []
        
        for i, (scale, freq) in enumerate(zip(cwt_result.scales, cwt_result.frequencies)):
            scale_power = power[i, :]
            if np.max(scale_power) > threshold * max_power:
                dominant_scales.append(scale)
                dominant_frequencies.append(freq)
                peak_times.append(cwt_result.time_index[np.argmax(scale_power)])
        
        return {
            'dominant_scales': dominant_scales,
            'dominant_frequencies': dominant_frequencies,
            'peak_times': peak_times,
            'threshold_used': threshold
        }
    
    def analyze(
        self,
        data: Union[np.ndarray, pd.Series],
        method: str = 'both'
    ) -> Dict:
        """
        Comprehensive wavelet analysis.
        
        Parameters
        ----------
        data : array-like
            Input time series
        method : str
            'cwt', 'dwt', or 'both'
            
        Returns
        -------
        dict
            Complete analysis results
        """
        if isinstance(data, pd.Series):
            data = data.values
        
        results = {'input_length': len(data)}
        
        if method in ['cwt', 'both']:
            cwt = self.continuous_wavelet_transform(data)
            patterns = self.detect_scale_patterns(cwt)
            results['cwt'] = {
                'result': cwt,
                'patterns': patterns,
                'dominant_period': 1/patterns['dominant_frequencies'][0] if patterns['dominant_frequencies'] else None
            }
        
        if method in ['dwt', 'both']:
            dwt = self.discrete_wavelet_transform(data)
            results['dwt'] = dwt
            
            # Compute energy at each level
            total_energy = np.sum(data**2)
            energies = {}
            for i, detail in enumerate(dwt['details']):
                energies[f'level_{i}_detail'] = np.sum(detail**2) / total_energy
            energies['approximation'] = np.sum(dwt['approximation']**2) / total_energy
            results['dwt']['energy_distribution'] = energies
        
        return results


# Convenience function for quick analysis
def wavelet_analysis(
    data: np.ndarray,
    wavelet: str = 'morlet',
    sampling_rate: float = 1.0
) -> Dict:
    """
    Quick wavelet analysis of time series data.
    
    Parameters
    ----------
    data : np.ndarray
        Input time series
    wavelet : str
        Wavelet type
    sampling_rate : float
        Samples per unit time
        
    Returns
    -------
    dict
        Analysis results including CWT and DWT decompositions
    """
    lens = WaveletLens(wavelet=wavelet, sampling_rate=sampling_rate)
    return lens.analyze(data)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create synthetic signal with multiple frequency components
    t = np.linspace(0, 10, 1000)
    signal = (np.sin(2 * np.pi * 0.5 * t) +           # 0.5 Hz
              0.5 * np.sin(2 * np.pi * 2 * t) +        # 2 Hz
              0.3 * np.sin(2 * np.pi * 5 * t) +        # 5 Hz
              0.2 * np.random.randn(len(t)))           # Noise
    
    # Analyze
    lens = WaveletLens(wavelet='morlet', sampling_rate=100)
    results = lens.analyze(signal)
    
    print("Wavelet Analysis Complete")
    print(f"Input length: {results['input_length']}")
    print(f"\nCWT Analysis:")
    print(f"  Detected {len(results['cwt']['patterns']['dominant_frequencies'])} dominant frequencies")
    if results['cwt']['dominant_period']:
        print(f"  Primary period: {results['cwt']['dominant_period']:.2f} time units")
    
    print(f"\nDWT Analysis (Haar):")
    print(f"  Decomposition levels: {results['dwt']['levels']}")
    print("  Energy distribution:")
    for level, energy in results['dwt']['energy_distribution'].items():
        print(f"    {level}: {energy*100:.1f}%")
