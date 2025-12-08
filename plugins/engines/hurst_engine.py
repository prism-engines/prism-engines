"""
Hurst Exponent Engine
=====================

Measures long-term memory and persistence in time series.
Key component of cross-domain rhythm analysis.

Interpretation:
- H > 0.5: Trending/persistent (momentum)
- H = 0.5: Random walk (no memory)
- H < 0.5: Mean-reverting (anti-persistent)

Methods:
- R/S (Rescaled Range): Classic, robust
- DFA (Detrended Fluctuation Analysis): Handles non-stationarity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HurstResult:
    """Result from Hurst analysis."""
    indicator: str
    hurst_rs: float          # R/S method
    hurst_dfa: Optional[float]  # DFA method (if computed)
    interpretation: str      # 'trending', 'random_walk', 'mean_reverting'
    confidence: float        # 0-1 confidence score


class HurstEngine:
    """
    Hurst Exponent Analysis Engine.
    
    Measures persistence/anti-persistence in time series,
    revealing whether indicators trend or mean-revert.
    """
    
    # Plugin metadata
    METADATA = {
        'name': 'Hurst Exponent Engine',
        'version': '1.0.0',
        'description': 'Long-term memory analysis via Hurst exponent',
        'author': 'PRISM Project',
        'tags': ['persistence', 'memory', 'fractal', 'cross-domain'],
        'dependencies': ['numpy', 'pandas']
    }
    
    def __init__(self, min_window: int = 20, max_window: int = 252):
        """
        Initialize Hurst engine.
        
        Args:
            min_window: Minimum window for R/S calculation
            max_window: Maximum window for R/S calculation
        """
        self.min_window = min_window
        self.max_window = max_window
    
    def compute_hurst_rs(self, series: np.ndarray) -> Tuple[float, float]:
        """
        Compute Hurst exponent using Rescaled Range (R/S) method.
        
        Args:
            series: 1D array of values
            
        Returns:
            Tuple of (hurst_exponent, r_squared)
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]
        
        n = len(series)
        if n < self.min_window * 2:
            return 0.5, 0.0  # Default to random walk
        
        # Generate window sizes (powers of 2 work well)
        max_k = min(int(np.log2(n / 4)), int(np.log2(self.max_window)))
        min_k = max(int(np.log2(self.min_window)), 2)
        
        if max_k <= min_k:
            return 0.5, 0.0
        
        window_sizes = [2**k for k in range(min_k, max_k + 1)]
        rs_values = []
        
        for window in window_sizes:
            if window > n // 2:
                continue
                
            # Number of windows
            n_windows = n // window
            rs_list = []
            
            for i in range(n_windows):
                start = i * window
                end = start + window
                subseries = series[start:end]
                
                # Mean-adjusted series
                mean_val = np.mean(subseries)
                adjusted = subseries - mean_val
                
                # Cumulative deviation
                cumdev = np.cumsum(adjusted)
                
                # Range
                R = np.max(cumdev) - np.min(cumdev)
                
                # Standard deviation
                S = np.std(subseries, ddof=1)
                
                if S > 0:
                    rs_list.append(R / S)
            
            if rs_list:
                rs_values.append((window, np.mean(rs_list)))
        
        if len(rs_values) < 3:
            return 0.5, 0.0
        
        # Log-log regression
        log_n = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        # Linear regression
        slope, intercept = np.polyfit(log_n, log_rs, 1)
        
        # R-squared
        predicted = slope * log_n + intercept
        ss_res = np.sum((log_rs - predicted) ** 2)
        ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Clamp Hurst to valid range
        hurst = np.clip(slope, 0.0, 1.0)
        
        return hurst, r_squared
    
    def compute_hurst_dfa(self, series: np.ndarray, order: int = 1) -> float:
        """
        Compute Hurst exponent using Detrended Fluctuation Analysis.
        
        More robust for non-stationary series.
        
        Args:
            series: 1D array of values
            order: Polynomial order for detrending (1=linear)
            
        Returns:
            Hurst exponent estimate
        """
        series = np.asarray(series)
        series = series[~np.isnan(series)]
        
        n = len(series)
        if n < 100:
            return 0.5
        
        # Integrate the series (cumulative sum of deviations from mean)
        mean_val = np.mean(series)
        integrated = np.cumsum(series - mean_val)
        
        # Generate window sizes
        min_window = max(10, order + 2)
        max_window = n // 4
        
        if max_window <= min_window:
            return 0.5
        
        # Use logarithmically spaced windows
        n_windows = min(20, int(np.log2(max_window / min_window)) + 1)
        window_sizes = np.unique(np.logspace(
            np.log10(min_window), 
            np.log10(max_window), 
            n_windows
        ).astype(int))
        
        fluctuations = []
        
        for window in window_sizes:
            n_segments = n // window
            if n_segments < 1:
                continue
            
            F_squared = []
            
            for i in range(n_segments):
                start = i * window
                end = start + window
                segment = integrated[start:end]
                
                # Fit polynomial trend
                x = np.arange(window)
                coeffs = np.polyfit(x, segment, order)
                trend = np.polyval(coeffs, x)
                
                # Detrended fluctuation
                detrended = segment - trend
                F_squared.append(np.mean(detrended ** 2))
            
            if F_squared:
                fluctuations.append((window, np.sqrt(np.mean(F_squared))))
        
        if len(fluctuations) < 3:
            return 0.5
        
        # Log-log regression
        log_n = np.log([x[0] for x in fluctuations])
        log_f = np.log([x[1] for x in fluctuations])
        
        slope, _ = np.polyfit(log_n, log_f, 1)
        
        return np.clip(slope, 0.0, 1.0)
    
    def interpret_hurst(self, h: float) -> str:
        """Interpret Hurst exponent value."""
        if h > 0.65:
            return 'strongly_trending'
        elif h > 0.55:
            return 'trending'
        elif h > 0.45:
            return 'random_walk'
        elif h > 0.35:
            return 'mean_reverting'
        else:
            return 'strongly_mean_reverting'
    
    def analyze(self, 
                data: pd.DataFrame,
                use_dfa: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze all indicators in panel.
        
        Args:
            data: DataFrame with indicators as columns
            use_dfa: Whether to also compute DFA estimates
            
        Returns:
            Dictionary with hurst_values, summary, interpretations
        """
        results = {}
        hurst_values = {}
        interpretations = {}
        
        for col in data.columns:
            series = data[col].dropna().values
            
            if len(series) < self.min_window * 2:
                continue
            
            # R/S method
            h_rs, r2 = self.compute_hurst_rs(series)
            
            # DFA method (optional)
            h_dfa = None
            if use_dfa and len(series) >= 100:
                h_dfa = self.compute_hurst_dfa(series)
            
            # Use average if both available
            if h_dfa is not None:
                h_combined = (h_rs + h_dfa) / 2
            else:
                h_combined = h_rs
            
            hurst_values[col] = round(h_combined, 4)
            interpretations[col] = self.interpret_hurst(h_combined)
            
            results[col] = HurstResult(
                indicator=col,
                hurst_rs=round(h_rs, 4),
                hurst_dfa=round(h_dfa, 4) if h_dfa else None,
                interpretation=self.interpret_hurst(h_combined),
                confidence=round(r2, 4)
            )
        
        # Summary statistics
        h_vals = list(hurst_values.values())
        summary = {
            'mean_hurst': round(np.mean(h_vals), 4) if h_vals else 0.5,
            'std_hurst': round(np.std(h_vals), 4) if h_vals else 0.0,
            'trending_count': sum(1 for h in h_vals if h > 0.55),
            'random_walk_count': sum(1 for h in h_vals if 0.45 <= h <= 0.55),
            'mean_reverting_count': sum(1 for h in h_vals if h < 0.45),
            'indicators_analyzed': len(h_vals)
        }
        
        # Regime signals
        avg_hurst = summary['mean_hurst']
        if avg_hurst > 0.6:
            regime_signal = 'trending_regime'
        elif avg_hurst < 0.4:
            regime_signal = 'mean_reverting_regime'
        else:
            regime_signal = 'neutral_regime'
        
        return {
            'hurst_values': hurst_values,
            'interpretations': interpretations,
            'detailed_results': {k: vars(v) for k, v in results.items()},
            'summary': summary,
            'regime_signal': regime_signal
        }
    
    def rank_indicators(self, 
                        data: pd.DataFrame,
                        by: str = 'deviation_from_random') -> List[Tuple[str, float]]:
        """
        Rank indicators by Hurst characteristics.
        
        Args:
            data: DataFrame with indicators
            by: Ranking method ('deviation_from_random', 'trending', 'mean_reverting')
            
        Returns:
            List of (indicator, score) tuples, sorted
        """
        results = self.analyze(data, use_dfa=False)
        hurst_values = results['hurst_values']
        
        if by == 'deviation_from_random':
            # How far from 0.5
            ranked = [(k, abs(v - 0.5)) for k, v in hurst_values.items()]
        elif by == 'trending':
            # Higher is more trending
            ranked = [(k, v) for k, v in hurst_values.items()]
        elif by == 'mean_reverting':
            # Lower is more mean-reverting
            ranked = [(k, 1 - v) for k, v in hurst_values.items()]
        else:
            ranked = [(k, v) for k, v in hurst_values.items()]
        
        return sorted(ranked, key=lambda x: x[1], reverse=True)


# =============================================================================
# Convenience function for quick analysis
# =============================================================================

def compute_hurst(series: np.ndarray) -> float:
    """Quick Hurst computation for a single series."""
    engine = HurstEngine()
    h, _ = engine.compute_hurst_rs(series)
    return h


# =============================================================================
# CLI / Standalone Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hurst Exponent Engine - Test")
    print("=" * 60)
    
    np.random.seed(42)
    n = 1000
    
    # Create test series with known properties
    # 1. Trending series (H > 0.5)
    trending = np.cumsum(np.random.randn(n) + 0.1)
    
    # 2. Random walk (H ≈ 0.5)
    random_walk = np.cumsum(np.random.randn(n))
    
    # 3. Mean-reverting series (H < 0.5)
    mean_reverting = np.zeros(n)
    for i in range(1, n):
        mean_reverting[i] = -0.5 * mean_reverting[i-1] + np.random.randn()
    
    panel = pd.DataFrame({
        'trending': trending,
        'random_walk': random_walk,
        'mean_reverting': mean_reverting
    })
    
    print(f"\nTest panel: {panel.shape[0]} rows, {panel.shape[1]} columns")
    
    # Run analysis
    engine = HurstEngine()
    results = engine.analyze(panel)
    
    print("\n📊 Hurst Values:")
    for indicator, h in results['hurst_values'].items():
        interp = results['interpretations'][indicator]
        print(f"   {indicator}: H={h:.3f} ({interp})")
    
    print(f"\n📈 Summary:")
    for k, v in results['summary'].items():
        print(f"   {k}: {v}")
    
    print(f"\n🎯 Regime Signal: {results['regime_signal']}")
    
    # Verification
    print("\n" + "-" * 60)
    print("VERIFICATION:")
    
    h = results['hurst_values']
    if h['trending'] > 0.55:
        print("✅ Trending series correctly identified (H > 0.55)")
    else:
        print(f"❌ Trending series: expected H > 0.55, got {h['trending']}")
    
    if 0.4 < h['random_walk'] < 0.6:
        print("✅ Random walk correctly identified (0.4 < H < 0.6)")
    else:
        print(f"⚠️ Random walk: expected ~0.5, got {h['random_walk']}")
    
    if h['mean_reverting'] < 0.45:
        print("✅ Mean-reverting series correctly identified (H < 0.45)")
    else:
        print(f"⚠️ Mean-reverting: expected H < 0.45, got {h['mean_reverting']}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
