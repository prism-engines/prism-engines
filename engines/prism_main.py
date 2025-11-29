"""
VCF Main Integration Module
============================

Complete Vector Coherence Framework integration.

This module provides a high-level interface to the entire VCF analysis pipeline:
1. Data loading and validation
2. Normalization (dual-input architecture)
3. Coherence analysis (phase relationships)
4. Geometric analysis (regime detection)
5. Output generation

Quick Start Example:
-------------------
>>> from prism_main import VCFPipeline
>>> import pandas as pd
>>>
>>> # Load your market data
>>> data = {
...     'GDP': gdp_series,
...     'SP500': sp500_series,
...     'Treasury10Y': yield_series
... }
>>>
>>> # Run complete analysis
>>> pipeline = VCFPipeline()
>>> results = pipeline.run_analysis(data)
>>>
>>> # Access results
>>> print(results['regimes'])
>>> print(results['coherence_matrix'])
>>> results['state_matrix'].to_csv('output.csv')
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import warnings

# Import VCF modules
from .normalization_engine import VCFNormalizer, create_state_matrix
from .coherence_engine import CoherenceEngine, PhaseLockingAnalysis
from .geometry_engine import GeometricAnalyzer, RegimeDetector


class VCFPipeline:
    """
    Complete VCF analysis pipeline.
    
    This class orchestrates the entire VCF framework, from raw data
    to final regime classifications and coherence metrics.
    """
    
    def __init__(self,
                 ma_window: int = 12,
                 roc_window: int = 1,
                 sampling_freq: float = 12.0):
        """
        Initialize VCF pipeline.
        
        Parameters:
        -----------
        ma_window : int
            Moving average window for normalization (default 12 months)
        roc_window : int
            Rate of change window for momentum (default 1 month)
        sampling_freq : float
            Sampling frequency for coherence analysis (12 = monthly)
        """
        self.normalizer = VCFNormalizer(ma_window, roc_window)
        self.coherence = CoherenceEngine(sampling_freq)
        self.geometry = GeometricAnalyzer()
        self.regime_detector = RegimeDetector(self.geometry)
        
        self.results = {}
        self.state_matrix = None
        
    def validate_data(self, market_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Validate and clean input market data.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary of {name: time_series}
            
        Returns:
        --------
        dict: Cleaned market data
        
        Checks:
        -------
        - All series are pandas Series
        - All have datetime index
        - Minimum length requirement (>= 24 observations)
        - Not too many missing values (< 50%)
        - No duplicate indices
        """
        cleaned = {}
        
        for name, series in market_data.items():
            # Check type
            if not isinstance(series, pd.Series):
                warnings.warn(f"Converting {name} to Series")
                series = pd.Series(series)
            
            # Check index
            if not isinstance(series.index, pd.DatetimeIndex):
                warnings.warn(f"{name} does not have datetime index, attempting conversion")
                try:
                    series.index = pd.to_datetime(series.index)
                except:
                    raise ValueError(f"Cannot convert {name} index to datetime")
            
            # Check length
            if len(series) < 24:
                warnings.warn(f"Skipping {name}: too short ({len(series)} < 24)")
                continue
            
            # Check missing data
            missing_pct = series.isna().sum() / len(series)
            if missing_pct > 0.5:
                warnings.warn(f"Skipping {name}: too many NaNs ({missing_pct:.1%})")
                continue
            
            # Check for duplicates
            if series.index.duplicated().any():
                warnings.warn(f"Removing {series.index.duplicated().sum()} duplicates from {name}")
                series = series[~series.index.duplicated(keep='first')]
            
            # Sort by date
            series = series.sort_index()
            
            cleaned[name] = series
        
        if len(cleaned) == 0:
            raise ValueError("No valid series after validation")
        
        return cleaned
    
    def normalize(self, market_data: Dict[str, pd.Series],
                 method: str = 'dual_input') -> pd.DataFrame:
        """
        Normalize market data into VCF state matrix.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary of {name: time_series}
        method : str
            Normalization method ('dual_input' recommended)
            
        Returns:
        --------
        pd.DataFrame: Normalized state matrix
        """
        # Validate data first
        clean_data = self.validate_data(market_data)
        
        # Create state matrix
        if method == 'dual_input':
            state = create_state_matrix(clean_data, self.normalizer)
        else:
            df = pd.DataFrame(clean_data)
            state = self.normalizer.batch_normalize(df, method=method)
        
        self.state_matrix = state
        return state
    
    def compute_coherence(self, state_matrix: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compute coherence metrics.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame, optional
            If None, uses self.state_matrix
            
        Returns:
        --------
        dict with:
            - coherence_matrix: Pairwise PLV matrix
            - kuramoto_order: Time series of Kuramoto order parameter
            - dynamic_coherence: Rolling coherence metrics
        """
        if state_matrix is None:
            if self.state_matrix is None:
                raise ValueError("No state matrix available. Run normalize() first.")
            state_matrix = self.state_matrix
        
        # Compute coherence matrix (pairwise PLV)
        coh_matrix = self.coherence.coherence_matrix(state_matrix, method='plv')
        
        # Compute Kuramoto order parameter
        phases = state_matrix.apply(lambda x: self.coherence.hilbert_phase(x), axis=0)
        kuramoto = self.coherence.kuramoto_order_parameter(phases)
        
        # Compute dynamic coherence
        dynamic_coh = self.coherence.dynamic_coherence(state_matrix, window=24)
        
        results = {
            'coherence_matrix': coh_matrix,
            'kuramoto_order': kuramoto,
            'dynamic_coherence': dynamic_coh,
            'phases': phases
        }
        
        return results
    
    def compute_geometry(self, state_matrix: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compute geometric metrics.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame, optional
            If None, uses self.state_matrix
            
        Returns:
        --------
        dict with:
            - magnitude: Distance from equilibrium
            - rotation: Angular change rate
            - divergence: Distance from mean
            - velocity: State space velocity
            - pca_projection: Principal components
        """
        if state_matrix is None:
            if self.state_matrix is None:
                raise ValueError("No state matrix available. Run normalize() first.")
            state_matrix = self.state_matrix
        
        # Compute basic geometric quantities
        magnitude = self.geometry.magnitude(state_matrix)
        rotation = self.geometry.angular_rotation(state_matrix)
        divergence = self.geometry.divergence_from_mean(state_matrix)
        velocity = self.geometry.velocity(state_matrix)
        velocity_mag = np.sqrt((velocity ** 2).sum(axis=1))
        
        # PCA projection
        try:
            pca_proj, pca_model = self.geometry.pca_projection(state_matrix, n_components=3)
            explained_var = pca_model.explained_variance_ratio_
        except:
            pca_proj = None
            explained_var = None
            warnings.warn("PCA projection failed")
        
        results = {
            'magnitude': magnitude,
            'rotation': rotation,
            'divergence': divergence,
            'velocity_magnitude': velocity_mag,
            'velocity': velocity,
            'pca_projection': pca_proj,
            'pca_explained_variance': explained_var
        }
        
        return results
    
    def detect_regimes(self, state_matrix: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detect market regimes.
        
        Parameters:
        -----------
        state_matrix : pd.DataFrame, optional
            If None, uses self.state_matrix
            
        Returns:
        --------
        dict with:
            - regime_signals: DataFrame of all geometric signals
            - regimes: Series of regime labels
            - regime_changes: Boolean series marking transitions
        """
        if state_matrix is None:
            if self.state_matrix is None:
                raise ValueError("No state matrix available. Run normalize() first.")
            state_matrix = self.state_matrix
        
        # Compute regime signals
        signals = self.regime_detector.compute_regime_signals(state_matrix)
        
        # Classify regimes
        regimes = self.regime_detector.classify_regime(signals)
        
        # Detect changes
        changes = self.regime_detector.detect_regime_changes(regimes)
        
        results = {
            'regime_signals': signals,
            'regimes': regimes,
            'regime_changes': changes
        }
        
        return results
    
    def run_analysis(self, market_data: Dict[str, pd.Series],
                    normalize_method: str = 'dual_input') -> Dict:
        """
        Run complete VCF analysis pipeline.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary of {name: time_series}
        normalize_method : str
            Normalization method
            
        Returns:
        --------
        dict containing all analysis results:
            - state_matrix: Normalized data
            - coherence_matrix: Pairwise coherence
            - kuramoto_order: Global synchronization
            - magnitude, rotation, divergence: Geometric metrics
            - regimes: Regime classifications
            - regime_signals: All geometric signals
            
        This is the main entry point for VCF analysis.
        """
        print("VCF Analysis Pipeline")
        print("=" * 60)
        
        # Step 1: Normalize
        print("\n[1/4] Normalizing data...")
        state_matrix = self.normalize(market_data, method=normalize_method)
        print(f"  → Created {state_matrix.shape[1]}-dimensional state matrix")
        print(f"  → Time range: {state_matrix.index[0]} to {state_matrix.index[-1]}")
        
        # Step 2: Coherence
        print("\n[2/4] Computing coherence metrics...")
        coherence_results = self.compute_coherence(state_matrix)
        print(f"  → Mean PLV: {coherence_results['coherence_matrix'].values[np.triu_indices_from(coherence_results['coherence_matrix'].values, k=1)].mean():.3f}")
        print(f"  → Mean Kuramoto order: {coherence_results['kuramoto_order'].mean():.3f}")
        
        # Step 3: Geometry
        print("\n[3/4] Computing geometric metrics...")
        geometry_results = self.compute_geometry(state_matrix)
        print(f"  → Mean magnitude: {geometry_results['magnitude'].mean():.3f}")
        print(f"  → Mean rotation: {geometry_results['rotation'].mean():.3f} rad")
        
        # Step 4: Regimes
        print("\n[4/4] Detecting regimes...")
        regime_results = self.detect_regimes(state_matrix)
        print(f"  → Regime distribution:")
        for regime, count in regime_results['regimes'].value_counts().items():
            print(f"      {regime}: {count} ({count/len(regime_results['regimes'])*100:.1f}%)")
        
        # Combine all results
        results = {
            'state_matrix': state_matrix,
            **coherence_results,
            **geometry_results,
            **regime_results
        }
        
        self.results = results
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        
        return results
    
    def export_results(self, results: Optional[Dict] = None,
                      output_dir: str = './vcf_output') -> None:
        """
        Export all results to CSV files.
        
        Parameters:
        -----------
        results : dict, optional
            Results dictionary (if None, uses self.results)
        output_dir : str
            Output directory path
        """
        import os
        
        if results is None:
            if not self.results:
                raise ValueError("No results to export. Run analysis first.")
            results = self.results
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export DataFrames and Series
        exports = {
            'state_matrix.csv': results.get('state_matrix'),
            'coherence_matrix.csv': results.get('coherence_matrix'),
            'kuramoto_order.csv': results.get('kuramoto_order'),
            'dynamic_coherence.csv': results.get('dynamic_coherence'),
            'geometric_metrics.csv': pd.DataFrame({
                'magnitude': results.get('magnitude'),
                'rotation': results.get('rotation'),
                'divergence': results.get('divergence'),
                'velocity_magnitude': results.get('velocity_magnitude')
            }),
            'regime_signals.csv': results.get('regime_signals'),
            'regimes.csv': results.get('regimes'),
            'pca_projection.csv': results.get('pca_projection')
        }
        
        for filename, data in exports.items():
            if data is not None:
                filepath = os.path.join(output_dir, filename)
                data.to_csv(filepath)
                print(f"Exported: {filepath}")
        
        print(f"\nAll results exported to: {output_dir}")


def quick_analysis(market_data: Dict[str, pd.Series]) -> Dict:
    """
    Convenience function for quick VCF analysis.
    
    Parameters:
    -----------
    market_data : dict
        Dictionary of {name: time_series}
        
    Returns:
    --------
    dict: Complete analysis results
    
    Example:
    --------
    >>> data = {'GDP': gdp_series, 'SP500': sp500_series}
    >>> results = quick_analysis(data)
    >>> print(results['regimes'])
    """
    pipeline = VCFPipeline()
    results = pipeline.run_analysis(market_data)
    return results


# Testing and example
if __name__ == "__main__":
    print("VCF Main Integration Module - Test")
    print("=" * 60)
    
    # Create synthetic market data
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=120, freq='M')
    
    # Simulate different market sources
    gdp = pd.Series(
        100 + np.cumsum(np.random.randn(120) * 0.5),
        index=dates,
        name='GDP'
    )
    
    sp500 = pd.Series(
        1000 + np.cumsum(np.random.randn(120) * 20),
        index=dates,
        name='SP500'
    )
    
    treasury = pd.Series(
        3.5 + 2 * np.sin(np.linspace(0, 8*np.pi, 120)) + np.random.randn(120) * 0.3,
        index=dates,
        name='Treasury10Y'
    )
    
    vix = pd.Series(
        15 + 10 * np.abs(np.sin(np.linspace(0, 6*np.pi, 120))) + np.random.randn(120) * 2,
        index=dates,
        name='VIX'
    )
    
    market_data = {
        'GDP': gdp,
        'SP500': sp500,
        'Treasury10Y': treasury,
        'VIX': vix
    }
    
    # Run analysis
    print("\nRunning complete VCF analysis on synthetic data...")
    results = quick_analysis(market_data)
    
    print("\n" + "=" * 60)
    print("Sample Results:")
    print("-" * 60)
    print("\nState Matrix (first 5 rows):")
    print(results['state_matrix'].head())
    
    print("\nCoherence Matrix:")
    print(results['coherence_matrix'])
    
    print("\nRegime Summary:")
    print(results['regimes'].value_counts())
    
    print("\n" + "=" * 60)
    print("Integration test completed successfully!")
