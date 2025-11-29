"""
VCF Coherence Engine
====================

Advanced coherence and phase relationship analysis for Vector Coherence Framework.

Mathematical Framework:
----------------------
Coherence measures how synchronized different market signals are.
Unlike correlation (which only measures linear relationships),
coherence captures:
- Phase relationships (are signals in-sync or out-of-phase?)
- Frequency-specific alignment (do signals move together at certain timescales?)
- Dynamic synchronization (is alignment strengthening or weakening?)

Key Metrics Implemented:
------------------------
1. Phase Locking Value (PLV) - measures phase synchronization
2. Kuramoto Order Parameter - global synchronization metric
3. Spectral Coherence - frequency-domain alignment
4. Instantaneous Phase via Hilbert Transform
5. Phase Difference Time Series

Academic References:
-------------------
- Lachaux et al. (1999) - Phase Locking Value
- Kuramoto (1984) - Order parameter for coupled oscillators
- Pikovsky, Rosenblum, Kurths (2001) - Synchronization theory
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, List, Dict


class CoherenceEngine:
    """
    Comprehensive coherence analysis for VCF framework.
    
    Analyzes phase relationships and synchronization between
    normalized financial time series.
    """
    
    def __init__(self, sampling_freq: float = 12.0):
        """
        Initialize coherence engine.
        
        Parameters:
        -----------
        sampling_freq : float
            Sampling frequency (12 = monthly data, 252 = daily data)
        """
        self.sampling_freq = sampling_freq
    
    def hilbert_phase(self, series: pd.Series) -> pd.Series:
        """
        Extract instantaneous phase using Hilbert transform.
        
        The Hilbert transform creates an analytic signal from a real signal,
        allowing us to compute instantaneous phase and amplitude.
        
        Parameters:
        -----------
        series : pd.Series
            Input time series
            
        Returns:
        --------
        pd.Series of instantaneous phase angles (in radians, -π to π)
        
        Mathematical Foundation:
        -----------------------
        For signal x(t):
        1. Compute analytic signal: z(t) = x(t) + i*H[x(t)]
        2. Phase: φ(t) = arg(z(t)) = arctan(H[x]/x)
        
        where H[x] is the Hilbert transform of x.
        """
        # Remove NaNs
        clean = series.dropna()
        if len(clean) < 3:
            return pd.Series(np.nan, index=series.index)
        
        # Compute Hilbert transform
        analytic_signal = signal.hilbert(clean.values)
        
        # Extract phase
        phase = np.angle(analytic_signal)
        
        # Return as Series with original index
        result = pd.Series(phase, index=clean.index)
        return result.reindex(series.index)
    
    def phase_locking_value(self, series1: pd.Series, 
                           series2: pd.Series,
                           window: Optional[int] = None) -> float:
        """
        Compute Phase Locking Value (PLV) between two signals.
        
        PLV measures phase synchronization, ranging from 0 (no sync) to 1 (perfect sync).
        It's more robust than correlation for detecting nonlinear relationships.
        
        Parameters:
        -----------
        series1, series2 : pd.Series
            Input time series
        window : int, optional
            If provided, compute PLV over sliding window
            
        Returns:
        --------
        float (or pd.Series if window provided): PLV between 0 and 1
        
        Mathematical Formula:
        --------------------
        PLV = |⟨e^(i*Δφ(t))⟩|
        
        where:
        - Δφ(t) = φ1(t) - φ2(t) is the phase difference
        - ⟨⟩ denotes time average
        - | | is the absolute value (magnitude of complex number)
        
        Interpretation:
        --------------
        - PLV ≈ 1: Signals are phase-locked (synchronized)
        - PLV ≈ 0: Signals have independent phases (unsynchronized)
        - PLV ≈ 0.5: Intermediate coupling
        """
        # Get phases
        phase1 = self.hilbert_phase(series1)
        phase2 = self.hilbert_phase(series2)
        
        # Align series
        aligned = pd.DataFrame({'p1': phase1, 'p2': phase2}).dropna()
        if len(aligned) < 10:
            return np.nan
        
        # Compute phase difference
        phase_diff = aligned['p1'] - aligned['p2']
        
        if window is None:
            # Global PLV
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            return plv
        else:
            # Rolling PLV
            def compute_plv_window(x):
                if len(x) < 5:
                    return np.nan
                return np.abs(np.mean(np.exp(1j * x)))
            
            rolling_plv = phase_diff.rolling(window=window).apply(
                compute_plv_window, raw=True
            )
            return rolling_plv
    
    def kuramoto_order_parameter(self, phases: pd.DataFrame) -> pd.Series:
        """
        Compute Kuramoto order parameter for multiple oscillators.
        
        Measures global synchronization across N coupled oscillators.
        This tells us if the entire market is moving in sync or chaotically.
        
        Parameters:
        -----------
        phases : pd.DataFrame
            Each column is the instantaneous phase of one market signal
            
        Returns:
        --------
        pd.Series: Order parameter over time (0 = chaos, 1 = sync)
        
        Mathematical Formula:
        --------------------
        R(t) = |1/N Σ e^(i*φ_k(t))|
        
        where:
        - N = number of oscillators
        - φ_k(t) = phase of oscillator k at time t
        - | | = complex magnitude
        
        Physical Interpretation:
        -----------------------
        The Kuramoto model describes coupled oscillators (pendulums, neurons, markets).
        R = 1: All oscillators perfectly synchronized (market regime)
        R = 0: Complete disorder (transition/crisis)
        0 < R < 1: Partial synchronization (normal market)
        """
        N = phases.shape[1]
        
        # Compute complex exponentials for each phase
        exp_phases = np.exp(1j * phases.values)
        
        # Mean across oscillators for each time point
        mean_exp = np.nanmean(exp_phases, axis=1)
        
        # Order parameter is magnitude of mean
        order_param = np.abs(mean_exp)
        
        return pd.Series(order_param, index=phases.index)
    
    def spectral_coherence(self, series1: pd.Series,
                          series2: pd.Series,
                          nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spectral coherence between two signals.
        
        Measures frequency-specific correlation. Useful for finding which
        timescales (frequencies) show strongest alignment.
        
        Parameters:
        -----------
        series1, series2 : pd.Series
            Input time series
        nperseg : int, optional
            Length of each segment for Welch's method
            
        Returns:
        --------
        frequencies : np.ndarray
            Array of frequencies
        coherence : np.ndarray
            Coherence values (0 to 1) at each frequency
            
        Interpretation:
        --------------
        High coherence at low frequencies: Long-term alignment
        High coherence at high frequencies: Short-term co-movement
        Low coherence: Independent dynamics at that timescale
        """
        # Align and clean
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < 20:
            return np.array([]), np.array([])
        
        if nperseg is None:
            nperseg = min(len(df) // 4, 256)
        
        # Compute coherence using Welch's method
        freqs, coh = signal.coherence(
            df['s1'].values,
            df['s2'].values,
            fs=self.sampling_freq,
            nperseg=nperseg
        )
        
        return freqs, coh
    
    def phase_difference_series(self, series1: pd.Series,
                               series2: pd.Series) -> pd.Series:
        """
        Compute instantaneous phase difference between two signals.
        
        Parameters:
        -----------
        series1, series2 : pd.Series
            Input time series
            
        Returns:
        --------
        pd.Series: Phase difference over time (wrapped to -π to π)
        
        Interpretation:
        --------------
        Δφ ≈ 0: In phase (moving together)
        Δφ ≈ π: Out of phase (moving opposite)
        Δφ changing slowly: Phase-locked
        Δφ changing rapidly: Unlocked/independent
        """
        phase1 = self.hilbert_phase(series1)
        phase2 = self.hilbert_phase(series2)
        
        phase_diff = phase1 - phase2
        
        # Wrap to -π to π
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
        
        return phase_diff
    
    def coherence_matrix(self, state_matrix: pd.DataFrame,
                        method: str = 'plv') -> pd.DataFrame:
        """
        Compute pairwise coherence between all columns.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data (from VCFNormalizer)
        method : str
            'plv' - Phase Locking Value
            'correlation' - Traditional correlation (for comparison)
            
        Returns:
        --------
        pd.DataFrame: Symmetric matrix of coherence values
        
        Use Case:
        ---------
        Identifies which market signals are most synchronized.
        Can reveal:
        - Sector clustering
        - Leading/lagging relationships
        - Regime-dependent coupling
        """
        cols = state_matrix.columns
        n = len(cols)
        
        result = pd.DataFrame(index=cols, columns=cols, dtype=float)
        
        if method == 'plv':
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        result.iloc[i, j] = 1.0
                    else:
                        plv = self.phase_locking_value(
                            state_matrix.iloc[:, i],
                            state_matrix.iloc[:, j]
                        )
                        result.iloc[i, j] = plv
                        result.iloc[j, i] = plv
        
        elif method == 'correlation':
            result = state_matrix.corr()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return result
    
    def dynamic_coherence(self, state_matrix: pd.DataFrame,
                         window: int = 24) -> pd.DataFrame:
        """
        Compute rolling Kuramoto order parameter.
        
        Shows how global market synchronization evolves over time.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame
            Normalized market data
        window : int
            Rolling window size
            
        Returns:
        --------
        pd.DataFrame with columns:
            - order_parameter: Kuramoto R
            - mean_coherence: Average pairwise PLV
            
        Interpretation:
        --------------
        Rising order parameter: Market becoming more synchronized
        Falling order parameter: Market fragmenting/transitioning
        Sudden drops: Potential regime change
        """
        # Extract phases for all signals
        phases = state_matrix.apply(lambda x: self.hilbert_phase(x), axis=0)
        
        # Compute rolling order parameter
        order_params = []
        mean_cohs = []
        
        for i in range(window, len(state_matrix)):
            window_phases = phases.iloc[i-window:i]
            
            # Kuramoto order parameter
            N = window_phases.shape[1]
            exp_phases = np.exp(1j * window_phases.values)
            mean_exp = np.nanmean(exp_phases, axis=1)
            R = np.abs(np.mean(mean_exp))
            order_params.append(R)
            
            # Mean pairwise coherence
            n_cols = window_phases.shape[1]
            coh_sum = 0
            count = 0
            for ii in range(n_cols):
                for jj in range(ii+1, n_cols):
                    phase_diff = window_phases.iloc[:, ii] - window_phases.iloc[:, jj]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    coh_sum += plv
                    count += 1
            mean_coh = coh_sum / count if count > 0 else np.nan
            mean_cohs.append(mean_coh)
        
        result = pd.DataFrame({
            'order_parameter': order_params,
            'mean_coherence': mean_cohs
        }, index=state_matrix.index[window:])
        
        return result


class PhaseLockingAnalysis:
    """
    Advanced phase-locking analysis for specific signal pairs.
    
    Use this to study the relationship between two market signals in detail.
    """
    
    def __init__(self, series1: pd.Series, series2: pd.Series,
                 name1: str = "Signal1", name2: str = "Signal2"):
        """
        Initialize with two time series.
        
        Parameters:
        -----------
        series1, series2 : pd.Series
            Market signals to analyze
        name1, name2 : str
            Names for the signals
        """
        self.series1 = series1
        self.series2 = series2
        self.name1 = name1
        self.name2 = name2
        self.engine = CoherenceEngine()
        
    def full_analysis(self) -> Dict:
        """
        Perform complete phase-locking analysis.
        
        Returns:
        --------
        dict with:
            - phases: DataFrame of both phases
            - phase_diff: Phase difference time series
            - plv_global: Global PLV value
            - plv_rolling: Rolling PLV series
            - spectral_coh: Tuple of (frequencies, coherence)
        """
        # Extract phases
        phase1 = self.engine.hilbert_phase(self.series1)
        phase2 = self.engine.hilbert_phase(self.series2)
        
        phases_df = pd.DataFrame({
            f'{self.name1}_phase': phase1,
            f'{self.name2}_phase': phase2
        })
        
        # Phase difference
        phase_diff = self.engine.phase_difference_series(
            self.series1, self.series2
        )
        
        # Global PLV
        plv_global = self.engine.phase_locking_value(
            self.series1, self.series2
        )
        
        # Rolling PLV
        plv_rolling = self.engine.phase_locking_value(
            self.series1, self.series2, window=12
        )
        
        # Spectral coherence
        freqs, coh = self.engine.spectral_coherence(
            self.series1, self.series2
        )
        
        return {
            'phases': phases_df,
            'phase_diff': phase_diff,
            'plv_global': plv_global,
            'plv_rolling': plv_rolling,
            'spectral_coherence': (freqs, coh)
        }
    
    def detect_phase_slips(self, threshold: float = 1.0) -> pd.Series:
        """
        Detect sudden phase jumps (phase slips).
        
        Phase slips indicate moments when synchronization breaks down.
        In markets, these often coincide with regime changes or shocks.
        
        Parameters:
        -----------
        threshold : float
            Phase change (in radians) to count as slip
            
        Returns:
        --------
        pd.Series: Boolean series marking phase slip events
        """
        phase_diff = self.engine.phase_difference_series(
            self.series1, self.series2
        )
        
        # Compute rate of change of phase difference
        phase_diff_change = phase_diff.diff().abs()
        
        # Mark large changes as phase slips
        slips = phase_diff_change > threshold
        
        return slips


# Testing and example usage
if __name__ == "__main__":
    print("VCF Coherence Engine - Test Suite")
    print("=" * 60)
    
    # Create synthetic coupled oscillators
    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    
    # Two coupled signals with some phase lag
    freq1 = 0.1
    freq2 = 0.1
    phase_lag = np.pi / 4  # 45 degree lag
    
    signal1 = pd.Series(
        np.sin(2 * np.pi * freq1 * t) + 0.1 * np.random.randn(500),
        index=pd.date_range('2000-01-01', periods=500, freq='M'),
        name='Signal1'
    )
    
    signal2 = pd.Series(
        np.sin(2 * np.pi * freq2 * t + phase_lag) + 0.1 * np.random.randn(500),
        index=pd.date_range('2000-01-01', periods=500, freq='M'),
        name='Signal2'
    )
    
    # Initialize engine
    engine = CoherenceEngine()
    
    print("\n1. Testing Hilbert Phase Extraction")
    print("-" * 60)
    phase1 = engine.hilbert_phase(signal1)
    print(f"Phase 1 range: [{phase1.min():.3f}, {phase1.max():.3f}]")
    print(f"Phase 1 mean: {phase1.mean():.3f}")
    
    print("\n2. Testing Phase Locking Value")
    print("-" * 60)
    plv = engine.phase_locking_value(signal1, signal2)
    print(f"Global PLV: {plv:.3f}")
    print(f"Expected ~0.9 for coupled oscillators")
    
    print("\n3. Testing Kuramoto Order Parameter")
    print("-" * 60)
    phases_df = pd.DataFrame({
        's1': phase1,
        's2': engine.hilbert_phase(signal2)
    })
    order = engine.kuramoto_order_parameter(phases_df)
    print(f"Mean order parameter: {order.mean():.3f}")
    print(f"Order parameter std: {order.std():.3f}")
    
    print("\n4. Testing Spectral Coherence")
    print("-" * 60)
    freqs, coh = engine.spectral_coherence(signal1, signal2)
    print(f"Peak coherence: {coh.max():.3f} at frequency {freqs[coh.argmax()]:.3f}")
    
    print("\n5. Testing Phase-Locking Analysis")
    print("-" * 60)
    analysis = PhaseLockingAnalysis(signal1, signal2, "Sig1", "Sig2")
    results = analysis.full_analysis()
    print(f"Global PLV: {results['plv_global']:.3f}")
    print(f"Rolling PLV mean: {results['plv_rolling'].mean():.3f}")
    
    print("\n6. Testing Coherence Matrix")
    print("-" * 60)
    # Create multi-signal dataset
    signal3 = pd.Series(
        np.sin(2 * np.pi * 0.15 * t) + 0.1 * np.random.randn(500),
        index=signal1.index,
        name='Signal3'
    )
    state = pd.DataFrame({'S1': signal1, 'S2': signal2, 'S3': signal3})
    coh_matrix = engine.coherence_matrix(state, method='plv')
    print("\nCoherence Matrix:")
    print(coh_matrix)
    
    print("\n" + "=" * 60)
    print("All coherence tests completed successfully!")
