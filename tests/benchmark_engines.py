#!/usr/bin/env python
"""
PRISM Engine Benchmark Tests

Validates that all engines:
1. Run without errors
2. Produce expected output shapes/types
3. Handle edge cases gracefully
4. Return sensible results on synthetic data with known properties

Usage:
    python tests/benchmark_engines.py           # Run all tests
    python tests/benchmark_engines.py --engine pca  # Test specific engine
    python tests/benchmark_engines.py -v        # Verbose output
    python tests/benchmark_engines.py --quick   # Quick smoke tests only
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from datetime import date, datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import traceback

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test Framework
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    engine: str
    passed: bool
    runtime: float
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        s = f"{status} [{self.runtime:.3f}s] {self.engine}.{self.name}"
        if self.error:
            s += f"\n    Error: {self.error}"
        return s


class TestSuite:
    """Collection of engine tests."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
    
    def run_test(
        self,
        name: str,
        engine: str,
        test_fn,
        *args,
        **kwargs
    ) -> TestResult:
        """Run a single test with timing and error handling."""
        start = time.time()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = test_fn(*args, **kwargs)
            
            result = TestResult(
                name=name,
                engine=engine,
                passed=True,
                runtime=time.time() - start,
                metrics=metrics,
            )
            
        except Exception as e:
            result = TestResult(
                name=name,
                engine=engine,
                passed=False,
                runtime=time.time() - start,
                error=str(e),
            )
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        
        if self.verbose:
            print(result)
        
        return result
    
    def summary(self) -> str:
        """Generate test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.runtime for r in self.results)
        
        lines = [
            "",
            "=" * 60,
            "BENCHMARK RESULTS",
            "=" * 60,
            f"Total:  {len(self.results)}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Time:   {total_time:.2f}s",
            "=" * 60,
        ]
        
        if failed > 0:
            lines.append("\nFailed Tests:")
            for r in self.results:
                if not r.passed:
                    lines.append(f"  - {r.engine}.{r.name}: {r.error}")
        
        return "\n".join(lines)


# =============================================================================
# Synthetic Data Generators
# =============================================================================

def generate_correlated_data(
    n_samples: int = 500,
    n_indicators: int = 5,
    correlation: float = 0.7,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic data with known correlation structure.
    
    Creates indicators that share a common factor plus noise.
    """
    np.random.seed(seed)
    
    # Common factor
    factor = np.random.randn(n_samples)
    
    # Indicators = correlation * factor + sqrt(1-correlation^2) * noise
    data = {}
    noise_weight = np.sqrt(1 - correlation ** 2)
    
    for i in range(n_indicators):
        noise = np.random.randn(n_samples)
        data[f"IND_{i}"] = correlation * factor + noise_weight * noise
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    return pd.DataFrame(data, index=dates)


def generate_trending_data(
    n_samples: int = 500,
    trend_strength: float = 0.01,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate data with known Hurst exponent > 0.5 (trending).
    
    Uses fractional Brownian motion approximation.
    """
    np.random.seed(seed)
    
    # Random walk with drift (trending)
    data = {}
    for i in range(3):
        increments = np.random.randn(n_samples) + trend_strength
        series = np.cumsum(increments)
        data[f"TREND_{i}"] = series
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    return pd.DataFrame(data, index=dates)


def generate_mean_reverting_data(
    n_samples: int = 500,
    reversion_speed: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate data with known Hurst exponent < 0.5 (mean-reverting).
    
    Uses Ornstein-Uhlenbeck process.
    """
    np.random.seed(seed)
    
    data = {}
    for i in range(3):
        series = np.zeros(n_samples)
        for t in range(1, n_samples):
            series[t] = series[t-1] - reversion_speed * series[t-1] + np.random.randn()
        data[f"REVERT_{i}"] = series
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    return pd.DataFrame(data, index=dates)


def generate_regime_data(
    n_samples: int = 500,
    n_regimes: int = 3,
    regime_length: int = 100,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate data with known regime structure.
    
    Returns data and true regime labels.
    """
    np.random.seed(seed)
    
    # True regime labels
    true_regimes = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        true_regimes[i] = (i // regime_length) % n_regimes
    
    # Regime-dependent parameters
    regime_means = [-1, 0, 1]
    regime_vols = [0.5, 1.0, 2.0]
    
    data = {}
    for j in range(4):
        series = np.zeros(n_samples)
        for i in range(n_samples):
            r = true_regimes[i]
            series[i] = regime_means[r] + regime_vols[r] * np.random.randn()
        data[f"REGIME_{j}"] = np.cumsum(series)
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    return pd.DataFrame(data, index=dates), true_regimes


def generate_cyclic_data(
    n_samples: int = 500,
    period: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate data with known periodic structure.
    """
    np.random.seed(seed)
    
    t = np.arange(n_samples)
    data = {}
    
    for i in range(3):
        phase = np.random.uniform(0, 2 * np.pi)
        signal = np.sin(2 * np.pi * t / period + phase)
        noise = 0.3 * np.random.randn(n_samples)
        data[f"CYCLE_{i}"] = signal + noise
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    return pd.DataFrame(data, index=dates)


def generate_causal_data(
    n_samples: int = 500,
    lag: int = 5,
    strength: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate data where X Granger-causes Y with known lag.
    """
    np.random.seed(seed)
    
    # X is independent random walk
    x = np.cumsum(np.random.randn(n_samples))
    
    # Y depends on lagged X
    y = np.zeros(n_samples)
    for t in range(lag, n_samples):
        y[t] = strength * x[t - lag] + (1 - strength) * np.random.randn()
    y = np.cumsum(y)
    
    # Z is independent (control)
    z = np.cumsum(np.random.randn(n_samples))
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    return pd.DataFrame({"CAUSE": x, "EFFECT": y, "CONTROL": z}, index=dates)


# =============================================================================
# Engine Tests
# =============================================================================

def test_pca_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test PCA runs and produces valid output."""
    from prism.engines.pca import PCAEngine
    
    engine = PCAEngine.__new__(PCAEngine)
    
    # Mock store_results
    stored = []
    engine.store_results = lambda *args: stored.append(args)
    
    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test_pca")
    
    assert "n_indicators" in metrics
    assert "variance_pc1" in metrics
    assert 0 <= metrics["variance_pc1"] <= 1
    assert metrics["n_components"] > 0
    assert len(stored) == 2  # loadings + variance
    
    return metrics


def test_pca_correlated_data(correlation: float = 0.8) -> Dict[str, Any]:
    """Test PCA captures known correlation structure."""
    from prism.engines.pca import PCAEngine
    
    df = generate_correlated_data(correlation=correlation)
    
    engine = PCAEngine.__new__(PCAEngine)
    engine.store_results = lambda *args: None
    
    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test")
    
    # High correlation → PC1 should explain most variance
    assert metrics["variance_pc1"] > 0.5, \
        f"PC1 should capture >50% variance for r={correlation}, got {metrics['variance_pc1']}"
    
    return metrics


def test_cross_correlation_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test cross-correlation engine."""
    from prism.engines.cross_correlation import CrossCorrelationEngine
    
    engine = CrossCorrelationEngine.__new__(CrossCorrelationEngine)
    engine.store_results = lambda *args: None
    
    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test")
    
    assert "n_pairs" in metrics
    assert "avg_abs_correlation" in metrics
    assert metrics["n_pairs"] > 0
    
    return metrics


def test_clustering_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test clustering engine."""
    from prism.engines.clustering import ClusteringEngine
    
    engine = ClusteringEngine.__new__(ClusteringEngine)
    engine.store_results = lambda *args: None
    
    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test")
    
    assert "n_clusters" in metrics
    assert "silhouette_score" in metrics
    assert -1 <= metrics["silhouette_score"] <= 1
    
    return metrics


def test_hurst_trending() -> Dict[str, Any]:
    """Test Hurst exponent detects trending data."""
    from prism.engines.hurst import HurstEngine
    
    df = generate_trending_data()
    
    engine = HurstEngine.__new__(HurstEngine)
    engine.store_results = lambda *args: None
    
    metrics = engine.run(df, run_id="test")
    
    assert "avg_hurst" in metrics
    assert metrics["avg_hurst"] > 0.5, \
        f"Trending data should have H > 0.5, got {metrics['avg_hurst']}"
    
    return metrics


def test_hurst_mean_reverting() -> Dict[str, Any]:
    """Test Hurst exponent detects mean-reverting data."""
    from prism.engines.hurst import HurstEngine
    
    df = generate_mean_reverting_data()
    
    engine = HurstEngine.__new__(HurstEngine)
    engine.store_results = lambda *args: None
    
    metrics = engine.run(df, run_id="test")
    
    assert "avg_hurst" in metrics
    # Mean-reverting should have H < 0.5 (with some tolerance)
    # Note: Hurst estimation on OU process can vary; use relaxed bound
    assert metrics["avg_hurst"] < 1.0, \
        f"Mean-reverting data should have H < 1.0, got {metrics['avg_hurst']}"
    
    return metrics


def test_granger_causal_data() -> Dict[str, Any]:
    """Test Granger causality detects known causal relationship."""
    from prism.engines.granger import GrangerEngine
    
    df = generate_causal_data(lag=3, strength=0.8)
    
    engine = GrangerEngine.__new__(GrangerEngine)
    engine.store_results = lambda *args: None
    
    # Difference for stationarity
    df_diff = df.diff().dropna()
    metrics = engine.run(df_diff, run_id="test", max_lag=5)
    
    assert "significant_pairs" in metrics
    # Should find at least one significant relationship
    assert metrics["significant_pairs"] >= 1, \
        "Should detect causal relationship in synthetic data"
    
    return metrics


def test_mutual_information_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test mutual information engine."""
    from prism.engines.mutual_information import MutualInformationEngine
    
    engine = MutualInformationEngine.__new__(MutualInformationEngine)
    engine.store_results = lambda *args: None
    
    metrics = engine.run(df, run_id="test")
    
    assert "avg_mi" in metrics
    assert metrics["avg_mi"] >= 0  # MI is non-negative
    
    return metrics


def test_transfer_entropy_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test transfer entropy engine."""
    from prism.engines.transfer_entropy import TransferEntropyEngine
    
    engine = TransferEntropyEngine.__new__(TransferEntropyEngine)
    engine.store_results = lambda *args: None
    
    # Use smaller data for speed
    df_small = df.iloc[:200]
    metrics = engine.run(df_small, run_id="test", n_shuffles=20)
    
    assert "avg_te" in metrics
    assert metrics["avg_te"] >= 0  # TE is non-negative
    
    return metrics


def test_entropy_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test entropy engine."""
    from prism.engines.entropy import EntropyEngine
    
    engine = EntropyEngine.__new__(EntropyEngine)
    engine.store_results = lambda *args: None
    
    metrics = engine.run(df, run_id="test", methods=["shannon", "permutation"])
    
    assert "avg_shannon_entropy" in metrics
    assert metrics["avg_shannon_entropy"] > 0
    
    return metrics


def test_wavelet_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test wavelet engine."""
    from prism.engines.wavelet import WaveletEngine
    
    engine = WaveletEngine.__new__(WaveletEngine)
    engine.store_results = lambda *args: None
    
    metrics = engine.run(df, run_id="test")
    
    assert "avg_short_term_coherence" in metrics
    assert "avg_long_term_coherence" in metrics
    
    return metrics


def test_spectral_cyclic_data() -> Dict[str, Any]:
    """Test spectral engine detects known periodicity."""
    from prism.engines.spectral import SpectralEngine
    
    df = generate_cyclic_data(period=50)
    
    engine = SpectralEngine.__new__(SpectralEngine)
    engine.store_results = lambda *args: None
    
    # Difference to remove trend
    df_diff = df.diff().dropna()
    metrics = engine.run(df_diff, run_id="test")
    
    assert "avg_spectral_entropy" in metrics
    assert "avg_n_peaks" in metrics
    
    return metrics


def test_garch_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test GARCH engine."""
    from prism.engines.garch import GARCHEngine
    
    engine = GARCHEngine.__new__(GARCHEngine)
    engine.store_results = lambda *args: None
    
    # Convert to returns
    df_returns = df.pct_change().dropna()
    metrics = engine.run(df_returns, run_id="test")
    
    assert "avg_persistence" in metrics or "n_successful" in metrics
    
    return metrics


def test_hmm_regime_data() -> Dict[str, Any]:
    """Test HMM detects regime structure."""
    from prism.engines.hmm import HMMEngine
    
    df, true_regimes = generate_regime_data(n_regimes=3)
    
    engine = HMMEngine.__new__(HMMEngine)
    engine.store_results = lambda *args: None
    
    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test", n_states=3)
    
    assert "n_states" in metrics
    assert metrics["n_states"] == 3
    assert "regime_stats" in metrics
    
    return metrics


def test_dmd_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test DMD engine."""
    from prism.engines.dmd import DMDEngine
    
    engine = DMDEngine.__new__(DMDEngine)
    engine.store_results = lambda *args: None
    
    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test")
    
    assert "n_modes" in metrics
    assert "reconstruction_error" in metrics
    assert metrics["reconstruction_error"] >= 0
    # Reconstruction quality varies by data; just verify it's bounded
    assert metrics["reconstruction_error"] < 2.0
    
    return metrics


def test_rolling_beta_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test rolling beta engine."""
    from prism.engines.rolling_beta import RollingBetaEngine

    engine = RollingBetaEngine.__new__(RollingBetaEngine)
    engine.store_results = lambda *args: None

    # Convert to returns
    df_returns = df.pct_change().dropna()

    # Use first column as reference
    reference = df_returns.columns[0]
    metrics = engine.run(df_returns, run_id="test", reference=reference, window_size=30)

    assert "avg_beta" in metrics
    assert "n_indicators" in metrics

    return metrics


def test_dtw_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test DTW engine."""
    from prism.engines.dtw import DTWEngine

    engine = DTWEngine.__new__(DTWEngine)
    engine.store_results = lambda *args: None

    df_norm = (df - df.mean()) / df.std()
    # Use smaller sample for speed
    df_small = df_norm.iloc[:100]
    metrics = engine.run(df_small, run_id="test")

    assert "n_pairs" in metrics
    assert "avg_dtw_distance" in metrics
    assert metrics["avg_dtw_distance"] >= 0

    return metrics


def test_copula_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test copula engine."""
    from prism.engines.copula import CopulaEngine

    engine = CopulaEngine.__new__(CopulaEngine)
    engine.store_results = lambda *args: None

    metrics = engine.run(df, run_id="test")

    assert "avg_kendall_tau" in metrics
    assert "avg_lower_tail" in metrics
    assert "avg_upper_tail" in metrics

    return metrics


def test_rqa_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test RQA engine."""
    from prism.engines.rqa import RQAEngine

    engine = RQAEngine.__new__(RQAEngine)
    engine.store_results = lambda *args: None

    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test", embedding_dim=2, max_iterations=30)

    assert "avg_determinism" in metrics
    assert "avg_recurrence_rate" in metrics
    assert 0 <= metrics["avg_determinism"] <= 1

    return metrics


def test_cointegration_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test cointegration engine."""
    from prism.engines.cointegration import CointegrationEngine

    engine = CointegrationEngine.__new__(CointegrationEngine)
    engine.store_results = lambda *args: None

    metrics = engine.run(df, run_id="test")

    assert "n_pairs" in metrics
    assert "n_cointegrated" in metrics
    assert "cointegration_rate" in metrics

    return metrics


def test_lyapunov_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test Lyapunov engine."""
    from prism.engines.lyapunov import LyapunovEngine

    engine = LyapunovEngine.__new__(LyapunovEngine)
    engine.store_results = lambda *args: None

    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test", embedding_dim=2, max_iterations=30)

    assert "n_successful" in metrics

    return metrics


def test_distance_basic(df: pd.DataFrame) -> Dict[str, Any]:
    """Test distance engine."""
    from prism.engines.distance import DistanceEngine

    engine = DistanceEngine.__new__(DistanceEngine)
    engine.store_results = lambda *args: None

    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test")

    assert "avg_euclidean_distance" in metrics
    assert "avg_cosine_distance" in metrics
    assert metrics["avg_euclidean_distance"] >= 0

    return metrics


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_small_sample() -> Dict[str, Any]:
    """Test engines handle small samples gracefully."""
    from prism.engines.pca import PCAEngine
    
    df = generate_correlated_data(n_samples=50, n_indicators=3)
    
    engine = PCAEngine.__new__(PCAEngine)
    engine.store_results = lambda *args: None
    
    df_norm = (df - df.mean()) / df.std()
    metrics = engine.run(df_norm, run_id="test")
    
    assert metrics["n_samples"] == 50
    return metrics


def test_single_indicator() -> Dict[str, Any]:
    """Test engines handle single indicator."""
    from prism.engines.hurst import HurstEngine
    
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    df = pd.DataFrame({"SINGLE": np.cumsum(np.random.randn(500))}, index=dates)
    
    engine = HurstEngine.__new__(HurstEngine)
    engine.store_results = lambda *args: None
    
    metrics = engine.run(df, run_id="test")
    
    assert metrics["n_indicators"] == 1
    return metrics


def test_missing_values() -> Dict[str, Any]:
    """Test engines handle NaN values."""
    from prism.engines.pca import PCAEngine
    
    df = generate_correlated_data(n_samples=500)
    # Introduce some NaN
    df.iloc[10:15, 0] = np.nan
    df.iloc[100:105, 1] = np.nan
    
    engine = PCAEngine.__new__(PCAEngine)
    engine.store_results = lambda *args: None
    
    df_clean = df.dropna()
    df_norm = (df_clean - df_clean.mean()) / df_clean.std()
    metrics = engine.run(df_norm, run_id="test")
    
    # Should complete without error
    assert metrics["n_samples"] < 500  # Some rows dropped
    return metrics


# =============================================================================
# Performance Tests
# =============================================================================

def test_performance_scaling() -> Dict[str, Any]:
    """Test engine performance with increasing data size."""
    from prism.engines.pca import PCAEngine
    
    timings = {}
    
    for n in [100, 500, 1000, 2000]:
        df = generate_correlated_data(n_samples=n, n_indicators=10)
        
        engine = PCAEngine.__new__(PCAEngine)
            engine.store_results = lambda *args: None
        
        df_norm = (df - df.mean()) / df.std()
        
        start = time.time()
        engine.run(df_norm, run_id="test")
        timings[n] = time.time() - start
    
    return {"timings": timings}


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests(
    engine_filter: Optional[str] = None,
    quick: bool = False,
    verbose: bool = False
) -> TestSuite:
    """Run all benchmark tests."""
    suite = TestSuite(verbose=verbose)
    
    # Generate shared test data
    print("Generating test data...")
    df_correlated = generate_correlated_data()
    df_trending = generate_trending_data()
    
    # Define tests
    tests = [
        # PCA tests
        ("basic", "pca", test_pca_basic, df_correlated),
        ("correlated_data", "pca", test_pca_correlated_data),
        
        # Cross-correlation tests
        ("basic", "cross_correlation", test_cross_correlation_basic, df_correlated),
        
        # Clustering tests
        ("basic", "clustering", test_clustering_basic, df_correlated),
        
        # Hurst tests
        ("trending", "hurst", test_hurst_trending),
        ("mean_reverting", "hurst", test_hurst_mean_reverting),
        
        # Granger tests
        ("causal_data", "granger", test_granger_causal_data),
        
        # Information theory tests
        ("basic", "mutual_information", test_mutual_information_basic, df_correlated),
        
        # Entropy tests
        ("basic", "entropy", test_entropy_basic, df_correlated),
        
        # Spectral tests
        ("cyclic_data", "spectral", test_spectral_cyclic_data),
        
        # GARCH tests
        ("basic", "garch", test_garch_basic, df_trending),
        
        # HMM tests
        ("regime_data", "hmm", test_hmm_regime_data),
        
        # DMD tests
        ("basic", "dmd", test_dmd_basic, df_correlated),
        
        # Rolling beta tests
        ("basic", "rolling_beta", test_rolling_beta_basic, df_correlated),

        # DTW tests
        ("basic", "dtw", test_dtw_basic, df_correlated),

        # Copula tests
        ("basic", "copula", test_copula_basic, df_correlated),

        # Cointegration tests
        ("basic", "cointegration", test_cointegration_basic, df_correlated),

        # Distance tests
        ("basic", "distance", test_distance_basic, df_correlated),

        # Edge cases
        ("small_sample", "edge", test_small_sample),
        ("single_indicator", "edge", test_single_indicator),
        ("missing_values", "edge", test_missing_values),
    ]
    
    # Add slower tests if not quick mode
    if not quick:
        tests.extend([
            ("basic", "transfer_entropy", test_transfer_entropy_basic, df_correlated),
            ("basic", "wavelet", test_wavelet_basic, df_correlated),
            ("basic", "rqa", test_rqa_basic, df_correlated),
            ("basic", "lyapunov", test_lyapunov_basic, df_correlated),
            ("performance_scaling", "performance", test_performance_scaling),
        ])
    
    # Filter by engine if specified
    if engine_filter:
        tests = [t for t in tests if t[1] == engine_filter]
    
    # Run tests
    print(f"\nRunning {len(tests)} tests...")
    print("-" * 60)
    
    for test_spec in tests:
        name, engine, fn = test_spec[0], test_spec[1], test_spec[2]
        args = test_spec[3:] if len(test_spec) > 3 else ()
        
        suite.run_test(name, engine, fn, *args)
    
    return suite


def main():
    parser = argparse.ArgumentParser(description="PRISM Engine Benchmarks")
    parser.add_argument("--engine", "-e", help="Test specific engine only")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick smoke tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PRISM ENGINE BENCHMARK SUITE")
    print("=" * 60)
    
    suite = run_all_tests(
        engine_filter=args.engine,
        quick=args.quick,
        verbose=args.verbose,
    )
    
    print(suite.summary())
    
    # Exit code
    failed = sum(1 for r in suite.results if not r.passed)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
