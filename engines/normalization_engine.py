"""
VCF Normalization Module
========================

Advanced normalization framework for Vector Coherence Framework (VCF).
Handles the fundamental challenge of mixing oscillating and trending financial data.

Mathematical Framework:
----------------------
Traditional z-score normalization fails when combining:
- Oscillating data (yields, ratios) - bounded, mean-reverting
- Trending data (prices, indices) - unbounded, directional

Solution: Dual-Input Architecture
---------------------------------
For each market source, extract TWO signals:
1. POSITION: Where are we relative to trend? (Moving average ratio)
2. MOMENTUM: How fast are we changing? (Rate of change)

This creates a phase-space representation that captures both state and dynamics.

Academic References:
-------------------
- Fourier/Wavelet decomposition for harmonic analysis
- Hilbert transform for instantaneous phase
- Takens' embedding theorem for phase space reconstruction
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from typing import Tuple, Optional, Dict


class VCFNormalizer:
    """
    Advanced normalization for VCF framework.
    
    Transforms heterogeneous financial data into a unified geometric space
    suitable for vector analysis and coherence calculations.
    """
    
    def __init__(self, ma_window: int = 12, roc_window: int = 1):
        """
        Initialize normalizer with default windows.
        
        Parameters:
        -----------
        ma_window : int
            Window for moving average calculation (default 12 months)
        roc_window : int
            Window for rate of change calculation (default 1 month)
        """
        self.ma_window = ma_window
        self.roc_window = roc_window
        
    def dual_input_transform(self, series: pd.Series, 
                            name: str) -> pd.DataFrame:
        """
        Transform single time series into position + momentum signals.
        
        This is the CORE normalization that solves the trending/oscillating problem.
        
        Parameters:
        -----------
        series : pd.Series
            Raw financial time series
        name : str
            Name for the signal (used in column naming)
            
        Returns:
        --------
        pd.DataFrame with two columns:
            - {name}_position: ratio to moving average (where we are)
            - {name}_momentum: rate of change (how fast we're moving)
            
        Mathematical Justification:
        --------------------------
        Position = x(t) / MA(x, window)
        - Removes trend component
        - Oscillates around 1.0
        - Captures deviation from "normal" state
        
        Momentum = [x(t) - x(t-k)] / x(t-k)
        - Captures velocity in phase space
        - Independent of absolute level
        - Detects acceleration/deceleration
        """
        # Handle missing data
        if series.isna().sum() > len(series) * 0.3:
            raise ValueError(f"Series {name} has >30% missing data")
            
        series = series.ffill().bfill()
        
        # POSITION: Ratio to moving average
        ma = series.rolling(window=self.ma_window, min_periods=1).mean()
        position = series / ma
        
        # Handle edge case: if MA is zero or near-zero
        position = position.replace([np.inf, -np.inf], np.nan)
        position = position.fillna(1.0)
        
        # MOMENTUM: Rate of change
        momentum = series.pct_change(periods=self.roc_window)
        
        # Handle edge case: infinite percentage changes
        momentum = momentum.replace([np.inf, -np.inf], np.nan)
        momentum = momentum.fillna(0.0)
        
        # Combine into DataFrame
        result = pd.DataFrame({
            f'{name}_position': position,
            f'{name}_momentum': momentum
        }, index=series.index)
        
        return result
    
    def harmonic_normalize(self, series: pd.Series, 
                          n_components: int = 5) -> pd.DataFrame:
        """
        Normalize using harmonic decomposition (Fourier analysis).
        
        This method discovers the natural oscillatory structure in data
        BEFORE applying normalization, treating financial data as
        composed of multiple harmonic cycles.
        
        Parameters:
        -----------
        series : pd.Series
            Input time series
        n_components : int
            Number of Fourier components to retain
            
        Returns:
        --------
        pd.DataFrame with columns:
            - original: input series
            - trend: long-term component (DC + lowest frequencies)
            - cycle: cyclical component (retained harmonics)
            - noise: residual (removed harmonics)
            - harmonic_position: cycle / trend ratio
            
        Mathematical Foundation:
        -----------------------
        Any signal can be decomposed as:
        x(t) = Σ[A_k * cos(2πf_k*t + φ_k)]
        
        We separate:
        - Low frequencies (< f_threshold) → trend
        - Mid frequencies (primary cycles) → signal
        - High frequencies → noise
        """
        # Remove NaNs for FFT
        clean_series = series.dropna()
        if len(clean_series) < 12:
            raise ValueError("Need at least 12 observations for harmonic analysis")
        
        values = clean_series.values
        n = len(values)
        
        # Compute FFT
        fft_values = fft(values)
        frequencies = fftfreq(n)
        
        # Separate components by frequency
        # Trend: lowest frequency component (including DC)
        trend_mask = np.abs(frequencies) < (1.0 / (n / 2))  # Long cycles
        
        # Cycle: keep top n_components mid-frequency components
        power = np.abs(fft_values) ** 2
        power[trend_mask] = 0  # Exclude trend from cycle
        
        # Find n_components strongest frequencies
        top_indices = np.argsort(power)[-n_components:]
        cycle_mask = np.zeros(n, dtype=bool)
        cycle_mask[top_indices] = True
        
        # Reconstruct components
        trend_fft = fft_values.copy()
        trend_fft[~trend_mask] = 0
        trend = np.real(ifft(trend_fft))
        
        cycle_fft = fft_values.copy()
        cycle_fft[~cycle_mask] = 0
        cycle = np.real(ifft(cycle_fft))
        
        # Noise is everything else
        noise_fft = fft_values.copy()
        noise_fft[trend_mask | cycle_mask] = 0
        noise = np.real(ifft(noise_fft))
        
        # Create harmonic position: cycle/trend ratio
        # This is analogous to our position metric but frequency-aware
        harmonic_position = np.zeros(n)
        safe_trend = np.where(np.abs(trend) > 1e-6, trend, 1.0)
        harmonic_position = cycle / safe_trend
        
        # Build result DataFrame
        result = pd.DataFrame({
            'original': values,
            'trend': trend,
            'cycle': cycle,
            'noise': noise,
            'harmonic_position': harmonic_position
        }, index=clean_series.index)
        
        # Reindex to original series index (will have NaNs where original had NaNs)
        result = result.reindex(series.index)
        
        return result
    
    def robust_zscore(self, series: pd.Series, 
                     window: Optional[int] = None) -> pd.Series:
        """
        Robust z-score normalization using median and MAD.
        
        More resistant to outliers than standard z-score.
        Useful for financial data with extreme events.
        
        Parameters:
        -----------
        series : pd.Series
            Input series
        window : int, optional
            If provided, compute rolling robust z-score
            
        Returns:
        --------
        pd.Series of robust z-scores
        
        Formula:
        --------
        z = (x - median) / (1.4826 * MAD)
        
        where MAD = median(|x - median|)
        and 1.4826 is the consistency constant for normal distribution
        """
        if window is not None:
            # Rolling robust z-score
            def robust_z_single(x):
                if len(x) < 2:
                    return np.nan
                med = np.median(x)
                mad = np.median(np.abs(x - med))
                if mad < 1e-6:
                    return 0.0
                return (x.iloc[-1] - med) / (1.4826 * mad)
            
            return series.rolling(window=window).apply(robust_z_single, raw=False)
        else:
            # Global robust z-score
            med = series.median()
            mad = (series - med).abs().median()
            if mad < 1e-6:
                return pd.Series(0.0, index=series.index)
            return (series - med) / (1.4826 * mad)
    
    def batch_normalize(self, df: pd.DataFrame, 
                       method: str = 'dual_input') -> pd.DataFrame:
        """
        Normalize entire DataFrame of financial series.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Each column is a financial time series
        method : str
            'dual_input' - position + momentum (RECOMMENDED)
            'harmonic' - Fourier decomposition
            'robust_zscore' - Median-based z-score
            'standard_zscore' - Traditional z-score (NOT RECOMMENDED)
            
        Returns:
        --------
        pd.DataFrame with normalized values
        """
        if method == 'dual_input':
            # Apply dual-input to each column
            results = []
            for col in df.columns:
                try:
                    normalized = self.dual_input_transform(df[col], col)
                    results.append(normalized)
                except Exception as e:
                    print(f"Warning: Failed to normalize {col}: {e}")
                    continue
            return pd.concat(results, axis=1)
        
        elif method == 'harmonic':
            # Apply harmonic decomposition, return just harmonic_position
            results = []
            for col in df.columns:
                try:
                    decomp = self.harmonic_normalize(df[col])
                    results.append(decomp['harmonic_position'].rename(f'{col}_harmonic'))
                except Exception as e:
                    print(f"Warning: Failed harmonic normalize {col}: {e}")
                    continue
            return pd.concat(results, axis=1)
        
        elif method == 'robust_zscore':
            return df.apply(lambda x: self.robust_zscore(x), axis=0)
        
        elif method == 'standard_zscore':
            # Traditional z-score (included for comparison, but not recommended)
            return (df - df.mean()) / df.std(ddof=0)
        
        else:
            raise ValueError(f"Unknown method: {method}")


def create_state_matrix(market_data: Dict[str, pd.Series],
                       normalizer: Optional[VCFNormalizer] = None) -> pd.DataFrame:
    """
    High-level function to create VCF state matrix from raw market data.
    
    Parameters:
    -----------
    market_data : dict
        Dictionary mapping source names to time series
        Example: {'GDP': series1, 'SP500': series2, ...}
    normalizer : VCFNormalizer, optional
        Custom normalizer instance. If None, uses default.
        
    Returns:
    --------
    pd.DataFrame ready for geometric analysis
    
    Example:
    --------
    >>> market_data = {
    ...     'GDP': gdp_series,
    ...     'CPI': cpi_series,
    ...     'SP500': sp500_series,
    ...     'FEDFUNDS': fed_funds_series
    ... }
    >>> state_matrix = create_state_matrix(market_data)
    >>> # state_matrix now has 8 columns (4 sources × 2 signals each)
    """
    if normalizer is None:
        normalizer = VCFNormalizer()
    
    # Convert dict to DataFrame
    df = pd.DataFrame(market_data)
    
    # Apply dual-input normalization
    state = normalizer.batch_normalize(df, method='dual_input')
    
    # Sort by date and handle any remaining NaNs
    state = state.sort_index()
    state = state.ffill().bfill()
    
    # Drop any rows/columns that are still all NaN
    state = state.dropna(axis=1, how='all')
    state = state.dropna(axis=0, how='all')
    
    return state


# Example usage and testing
if __name__ == "__main__":
    print("VCF Normalization Module - Test Suite")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', periods=120, freq='M')
    
    # Trending series (like stock price)
    trend = np.linspace(100, 300, 120)
    cycle = 20 * np.sin(np.linspace(0, 8*np.pi, 120))
    noise = np.random.normal(0, 5, 120)
    trending_series = pd.Series(trend + cycle + noise, index=dates, name='SP500')
    
    # Oscillating series (like yield)
    oscillating_series = pd.Series(
        3.5 + 2 * np.sin(np.linspace(0, 10*np.pi, 120)) + np.random.normal(0, 0.3, 120),
        index=dates, 
        name='Treasury10Y'
    )
    
    # Test normalizer
    normalizer = VCFNormalizer()
    
    print("\n1. Testing Dual-Input Transform (Trending Data)")
    print("-" * 60)
    trend_normalized = normalizer.dual_input_transform(trending_series, 'SP500')
    print(trend_normalized.head())
    print(f"\nPosition mean: {trend_normalized['SP500_position'].mean():.3f}")
    print(f"Position std: {trend_normalized['SP500_position'].std():.3f}")
    
    print("\n2. Testing Dual-Input Transform (Oscillating Data)")
    print("-" * 60)
    osc_normalized = normalizer.dual_input_transform(oscillating_series, 'Yield')
    print(osc_normalized.head())
    print(f"\nPosition mean: {osc_normalized['Yield_position'].mean():.3f}")
    print(f"Position std: {osc_normalized['Yield_position'].std():.3f}")
    
    print("\n3. Testing Harmonic Decomposition")
    print("-" * 60)
    harmonic_result = normalizer.harmonic_normalize(trending_series, n_components=3)
    print(harmonic_result.head())
    
    print("\n4. Testing Batch Normalization")
    print("-" * 60)
    market_data = {
        'SP500': trending_series,
        'Treasury10Y': oscillating_series
    }
    state_matrix = create_state_matrix(market_data, normalizer)
    print(state_matrix.head())
    print(f"\nFinal state matrix shape: {state_matrix.shape}")
    print(f"Columns: {list(state_matrix.columns)}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
