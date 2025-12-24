"""
PRISM Engine Unit Tests (pytest compatible)

Run with:
    pytest tests/test_engines.py -v
    pytest tests/test_engines.py -k "pca" -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate basic sample data."""
    np.random.seed(42)
    n_samples = 500
    n_indicators = 5
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    data = {f"IND_{i}": np.cumsum(np.random.randn(n_samples)) for i in range(n_indicators)}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def correlated_data():
    """Generate correlated data."""
    np.random.seed(42)
    n_samples = 500
    n_indicators = 5
    correlation = 0.7
    
    factor = np.random.randn(n_samples)
    noise_weight = np.sqrt(1 - correlation ** 2)
    
    data = {}
    for i in range(n_indicators):
        noise = np.random.randn(n_samples)
        data[f"IND_{i}"] = correlation * factor + noise_weight * noise
    
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def returns_data(sample_data):
    """Convert sample data to returns."""
    return sample_data.pct_change().dropna()


@pytest.fixture
def normalized_data(sample_data):
    """Z-score normalized data."""
    return (sample_data - sample_data.mean()) / sample_data.std()


# =============================================================================
# Mock Engine Base
# =============================================================================

class MockEngineMixin:
    """Mixin to disable DB operations in tests."""

    def __init__(self):
        self._stored_results = []

    def store_results(self, table_name, df, run_id):
        self._stored_results.append((table_name, df, run_id))


# =============================================================================
# PCA Tests
# =============================================================================

class TestPCA:
    
    def test_pca_runs(self, normalized_data):
        """PCA completes without error."""
        from prism.engines.pca import PCAEngine
        
        class TestPCA(MockEngineMixin, PCAEngine):
            pass
        
        engine = TestPCA()
        metrics = engine.run(normalized_data, run_id="test")
        
        assert "variance_pc1" in metrics
        assert "n_components" in metrics
    
    def test_pca_variance_bounds(self, normalized_data):
        """PCA variance is between 0 and 1."""
        from prism.engines.pca import PCAEngine
        
        class TestPCA(MockEngineMixin, PCAEngine):
            pass
        
        engine = TestPCA()
        metrics = engine.run(normalized_data, run_id="test")
        
        assert 0 <= metrics["variance_pc1"] <= 1
    
    def test_pca_captures_correlation(self, correlated_data):
        """PCA captures known correlation structure."""
        from prism.engines.pca import PCAEngine
        
        class TestPCA(MockEngineMixin, PCAEngine):
            pass
        
        engine = TestPCA()
        df_norm = (correlated_data - correlated_data.mean()) / correlated_data.std()
        metrics = engine.run(df_norm, run_id="test")
        
        # High correlation → PC1 should explain >50% variance
        assert metrics["variance_pc1"] > 0.5
    
    def test_pca_stores_results(self, normalized_data):
        """PCA stores loadings and variance."""
        from prism.engines.pca import PCAEngine
        
        class TestPCA(MockEngineMixin, PCAEngine):
            pass
        
        engine = TestPCA()
        engine.run(normalized_data, run_id="test")
        
        stored_tables = [r[0] for r in engine._stored_results]
        assert "pca_loadings" in stored_tables
        assert "pca_variance" in stored_tables


# =============================================================================
# Hurst Tests
# =============================================================================

class TestHurst:
    
    def test_hurst_runs(self, sample_data):
        """Hurst completes without error."""
        from prism.engines.hurst import HurstEngine
        
        class TestHurst(MockEngineMixin, HurstEngine):
            pass
        
        engine = TestHurst()
        metrics = engine.run(sample_data, run_id="test")
        
        assert "avg_hurst" in metrics
    
    def test_hurst_bounds(self, sample_data):
        """Hurst exponent is between 0 and 1."""
        from prism.engines.hurst import HurstEngine
        
        class TestHurst(MockEngineMixin, HurstEngine):
            pass
        
        engine = TestHurst()
        metrics = engine.run(sample_data, run_id="test")
        
        assert 0 < metrics["avg_hurst"] < 1
    
    def test_hurst_trending_data(self):
        """Hurst detects trending data (H > 0.5)."""
        from prism.engines.hurst import HurstEngine
        
        class TestHurst(MockEngineMixin, HurstEngine):
            pass
        
        # Generate trending data
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "TREND": np.cumsum(np.random.randn(n) + 0.02)
        }, index=dates)
        
        engine = TestHurst()
        metrics = engine.run(df, run_id="test")
        
        assert metrics["avg_hurst"] > 0.5


# =============================================================================
# Clustering Tests
# =============================================================================

class TestClustering:
    
    def test_clustering_runs(self, normalized_data):
        """Clustering completes without error."""
        from prism.engines.clustering import ClusteringEngine
        
        class TestClustering(MockEngineMixin, ClusteringEngine):
            pass
        
        engine = TestClustering()
        metrics = engine.run(normalized_data, run_id="test")
        
        assert "n_clusters" in metrics
        assert "silhouette_score" in metrics
    
    def test_clustering_silhouette_bounds(self, normalized_data):
        """Silhouette score is between -1 and 1."""
        from prism.engines.clustering import ClusteringEngine
        
        class TestClustering(MockEngineMixin, ClusteringEngine):
            pass
        
        engine = TestClustering()
        metrics = engine.run(normalized_data, run_id="test")
        
        assert -1 <= metrics["silhouette_score"] <= 1
    
    def test_clustering_respects_n_clusters(self, normalized_data):
        """Clustering uses specified n_clusters."""
        from prism.engines.clustering import ClusteringEngine
        
        class TestClustering(MockEngineMixin, ClusteringEngine):
            pass
        
        engine = TestClustering()
        metrics = engine.run(normalized_data, run_id="test", n_clusters=3)
        
        assert metrics["n_clusters"] == 3


# =============================================================================
# Granger Tests
# =============================================================================

class TestGranger:
    
    def test_granger_runs(self, sample_data):
        """Granger completes without error."""
        from prism.engines.granger import GrangerEngine
        
        class TestGranger(MockEngineMixin, GrangerEngine):
            pass
        
        engine = TestGranger()
        df_diff = sample_data.diff().dropna()
        metrics = engine.run(df_diff, run_id="test")
        
        assert "n_tests" in metrics
        assert "significant_pairs" in metrics
    
    def test_granger_detects_causality(self):
        """Granger detects known causal relationship."""
        from prism.engines.granger import GrangerEngine
        
        class TestGranger(MockEngineMixin, GrangerEngine):
            pass
        
        # Generate causal data
        np.random.seed(42)
        n = 500
        lag = 3
        
        x = np.cumsum(np.random.randn(n))
        y = np.zeros(n)
        for t in range(lag, n):
            y[t] = 0.8 * x[t - lag] + 0.2 * np.random.randn()
        y = np.cumsum(y)
        
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame({"CAUSE": x, "EFFECT": y}, index=dates)
        df_diff = df.diff().dropna()
        
        engine = TestGranger()
        metrics = engine.run(df_diff, run_id="test", max_lag=5)
        
        assert metrics["significant_pairs"] >= 1


# =============================================================================
# HMM Tests
# =============================================================================

class TestHMM:
    
    def test_hmm_runs(self, normalized_data):
        """HMM completes without error."""
        from prism.engines.hmm import HMMEngine
        
        class TestHMM(MockEngineMixin, HMMEngine):
            pass
        
        engine = TestHMM()
        metrics = engine.run(normalized_data, run_id="test", n_states=2)
        
        assert "n_states" in metrics
        assert metrics["n_states"] == 2
    
    def test_hmm_regime_stats(self, normalized_data):
        """HMM produces regime statistics."""
        from prism.engines.hmm import HMMEngine
        
        class TestHMM(MockEngineMixin, HMMEngine):
            pass
        
        engine = TestHMM()
        metrics = engine.run(normalized_data, run_id="test", n_states=3)
        
        assert "regime_stats" in metrics
        assert len(metrics["regime_stats"]) == 3


# =============================================================================
# DMD Tests
# =============================================================================

class TestDMD:
    
    def test_dmd_runs(self, normalized_data):
        """DMD completes without error."""
        from prism.engines.dmd import DMDEngine
        
        class TestDMD(MockEngineMixin, DMDEngine):
            pass
        
        engine = TestDMD()
        metrics = engine.run(normalized_data, run_id="test")
        
        assert "n_modes" in metrics
        assert "reconstruction_error" in metrics
    
    def test_dmd_reconstruction(self, normalized_data):
        """DMD reconstruction error is reasonable."""
        from prism.engines.dmd import DMDEngine
        
        class TestDMD(MockEngineMixin, DMDEngine):
            pass
        
        engine = TestDMD()
        metrics = engine.run(normalized_data, run_id="test")
        
        # Reconstruction quality varies by data characteristics
        assert metrics["reconstruction_error"] < 2.0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    
    def test_small_sample(self):
        """Engines handle small samples."""
        from prism.engines.pca import PCAEngine
        
        class TestPCA(MockEngineMixin, PCAEngine):
            pass
        
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "A": np.random.randn(30),
            "B": np.random.randn(30),
        }, index=dates)
        df_norm = (df - df.mean()) / df.std()
        
        engine = TestPCA()
        metrics = engine.run(df_norm, run_id="test")
        
        assert metrics["n_samples"] == 30
    
    def test_two_indicators(self):
        """Engines handle minimum indicators."""
        from prism.engines.cross_correlation import CrossCorrelationEngine
        
        class TestCorr(MockEngineMixin, CrossCorrelationEngine):
            pass
        
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
        }, index=dates)
        df_norm = (df - df.mean()) / df.std()
        
        engine = TestCorr()
        metrics = engine.run(df_norm, run_id="test")
        
        assert metrics["n_pairs"] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    
    def test_engine_registry(self):
        """Engine registry works correctly."""
        from prism.engines import get_engine, list_engines, ENGINE_REGISTRY
        
        engines = list_engines()
        assert len(engines) >= 10
        
        for name in ENGINE_REGISTRY:
            engine = get_engine(name)
            assert engine.name == name
    
    def test_engine_phases(self):
        """Engines have valid phases."""
        from prism.engines import ENGINE_INFO
        
        valid_phases = {"unbound", "structure", "bounded"}
        
        for name, info in ENGINE_INFO.items():
            assert info["phase"] in valid_phases, f"{name} has invalid phase"
    
    def test_engine_priorities(self):
        """Engines have valid priorities."""
        from prism.engines import ENGINE_INFO
        
        for name, info in ENGINE_INFO.items():
            assert 1 <= info["priority"] <= 4, f"{name} has invalid priority"
