"""
Tests for Phases 8-12 Implementation
====================================

Comprehensive tests for:
- Phase 8: Statistical Foundation
- Phase 9: Null Models & Significance Testing
- Phase 10: Historical Validation
- Phase 11: Physics Formalism
- Phase 12: Multi-Domain Expansion
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestPhase8Stationarity:
    """Tests for stationarity testing module."""

    def test_stationarity_tester_import(self):
        """Test that StationarityTester can be imported."""
        from analysis.stationarity import StationarityTester
        tester = StationarityTester()
        assert tester is not None

    def test_adf_test_stationary(self):
        """Test ADF on stationary data."""
        from analysis.stationarity import StationarityTester

        np.random.seed(42)
        data = np.random.randn(500)  # White noise is stationary

        tester = StationarityTester()
        result = tester.adf_test(data)

        assert result.test_name == 'ADF'
        assert result.test_statistic is not None
        # White noise should be stationary
        assert result.is_stationary is True

    def test_kpss_test(self):
        """Test KPSS on stationary data."""
        from analysis.stationarity import StationarityTester

        np.random.seed(42)
        data = np.random.randn(500)

        tester = StationarityTester()
        result = tester.kpss_test(data)

        assert result.test_name == 'KPSS'
        assert result.test_statistic is not None

    def test_integration_order(self):
        """Test integration order determination."""
        from analysis.stationarity import StationarityTester

        np.random.seed(42)
        # Random walk (I(1))
        random_walk = np.cumsum(np.random.randn(500))

        tester = StationarityTester()
        order = tester.integration_order(random_walk)

        # Should be at least 1
        assert order >= 1


class TestPhase8Uncertainty:
    """Tests for uncertainty estimation module."""

    def test_uncertainty_estimator_import(self):
        """Test that UncertaintyEstimator can be imported."""
        from analysis.uncertainty import UncertaintyEstimator
        estimator = UncertaintyEstimator()
        assert estimator is not None

    def test_bootstrap_ci(self):
        """Test bootstrap confidence intervals."""
        from analysis.uncertainty import UncertaintyEstimator

        np.random.seed(42)
        data = np.random.randn(100) + 5  # Mean ~ 5

        estimator = UncertaintyEstimator(n_bootstrap=100)
        ci = estimator.bootstrap_ci(data, np.mean)

        # CI should contain true mean
        assert ci.lower < 5 < ci.upper
        assert ci.confidence_level == 0.95

    def test_block_bootstrap(self):
        """Test block bootstrap for time series."""
        from analysis.uncertainty import UncertaintyEstimator

        np.random.seed(42)
        data = np.random.randn(200)

        estimator = UncertaintyEstimator(n_bootstrap=100)
        ci = estimator.block_bootstrap_ci(data, np.mean, block_size=10)

        assert ci.lower is not None
        assert ci.upper is not None


class TestPhase8MultipleTesting:
    """Tests for multiple testing correction module."""

    def test_bonferroni(self):
        """Test Bonferroni correction."""
        from analysis.multiple_testing import bonferroni

        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        adjusted = bonferroni(p_values)

        # All p-values should increase
        for orig, adj in zip(p_values, adjusted):
            assert adj >= orig

    def test_benjamini_hochberg(self):
        """Test Benjamini-Hochberg FDR control."""
        from analysis.multiple_testing import benjamini_hochberg

        p_values = [0.001, 0.01, 0.02, 0.03, 0.5]
        adjusted = benjamini_hochberg(p_values)

        # Should be less conservative than Bonferroni
        from analysis.multiple_testing import bonferroni
        bonf_adj = bonferroni(p_values)

        for bh, bf in zip(adjusted, bonf_adj):
            assert bh <= bf


class TestPhase9Surrogates:
    """Tests for surrogate data generation."""

    def test_surrogate_generator_import(self):
        """Test that SurrogateGenerator can be imported."""
        from analysis.surrogates import SurrogateGenerator
        gen = SurrogateGenerator()
        assert gen is not None

    def test_phase_randomization(self):
        """Test phase randomization surrogates."""
        from analysis.surrogates import SurrogateGenerator

        np.random.seed(42)
        data = np.sin(np.linspace(0, 10*np.pi, 100)) + np.random.randn(100) * 0.1

        gen = SurrogateGenerator()
        surrogate = gen.phase_randomize(data)

        # Same length
        assert len(surrogate) == len(data)

        # Similar power spectrum (approximately)
        orig_power = np.abs(np.fft.rfft(data)) ** 2
        surr_power = np.abs(np.fft.rfft(surrogate)) ** 2

        # Correlation should be high
        corr = np.corrcoef(orig_power, surr_power)[0, 1]
        assert corr > 0.9

    def test_aaft_surrogate(self):
        """Test AAFT surrogates."""
        from analysis.surrogates import SurrogateGenerator

        np.random.seed(42)
        data = np.random.randn(100)

        gen = SurrogateGenerator()
        surrogate = gen.aaft(data)

        # Same length
        assert len(surrogate) == len(data)


class TestPhase9NullModels:
    """Tests for null model generation."""

    def test_null_model_generator_import(self):
        """Test that NullModelGenerator can be imported."""
        from analysis.null_models import NullModelGenerator
        gen = NullModelGenerator()
        assert gen is not None

    def test_white_noise(self):
        """Test white noise null model."""
        from analysis.null_models import NullModelGenerator

        np.random.seed(42)
        data = np.random.randn(500)

        gen = NullModelGenerator()
        null = gen.white_noise(data)

        # Same stats (approximately)
        assert abs(np.mean(null) - np.mean(data)) < 0.5
        assert abs(np.std(null) - np.std(data)) < 0.3

    def test_ar1_model(self):
        """Test AR(1) null model."""
        from analysis.null_models import NullModelGenerator

        np.random.seed(42)
        # Generate AR(1) process
        n = 500
        phi = 0.7
        data = np.zeros(n)
        for i in range(1, n):
            data[i] = phi * data[i-1] + np.random.randn()

        gen = NullModelGenerator()
        null = gen.ar1(data)

        assert len(null) == len(data)


class TestPhase10StressTests:
    """Tests for historical stress testing."""

    def test_stress_test_runner_import(self):
        """Test that StressTestRunner can be imported."""
        from validation.stress_tests import StressTestRunner
        runner = StressTestRunner()
        assert runner is not None

    def test_event_registry(self):
        """Test historical event registry."""
        from validation.stress_tests import HistoricalEventRegistry

        registry = HistoricalEventRegistry()
        events = registry.get_events()

        # Should have some events
        assert len(events) > 0


class TestPhase10TailAnalysis:
    """Tests for tail risk analysis."""

    def test_tail_analyzer_import(self):
        """Test that TailAnalyzer can be imported."""
        from validation.tail_analysis import TailAnalyzer
        analyzer = TailAnalyzer()
        assert analyzer is not None

    def test_var_calculation(self):
        """Test VaR calculation."""
        from validation.tail_analysis import TailAnalyzer

        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        analyzer = TailAnalyzer()
        var_95 = analyzer.historical_var(returns, alpha=0.05)

        # VaR should be negative (loss)
        assert var_95 < 0


class TestPhase11Geometry:
    """Tests for geometry module."""

    def test_metric_tensor_import(self):
        """Test that MetricTensor can be imported."""
        from theory.geometry import MetricTensor
        metric = MetricTensor(dim=3)
        assert metric is not None

    def test_distance_computation(self):
        """Test distance computation."""
        from theory.geometry import compute_distance

        x = np.array([0, 0, 0])
        y = np.array([1, 0, 0])

        dist = compute_distance(x, y)

        # Euclidean distance
        assert abs(dist - 1.0) < 0.001


class TestPhase11Ergodicity:
    """Tests for ergodicity analysis."""

    def test_ergodicity_analyzer_import(self):
        """Test that ErgodicityAnalyzer can be imported."""
        from theory.ergodicity import ErgodicityAnalyzer
        analyzer = ErgodicityAnalyzer()
        assert analyzer is not None

    def test_mean_ergodicity(self):
        """Test mean ergodicity check."""
        from theory.ergodicity import ErgodicityAnalyzer

        np.random.seed(42)
        # Stationary data should be ergodic
        data = np.random.randn(1000)

        analyzer = ErgodicityAnalyzer()
        result = analyzer.test_mean_ergodicity(data)

        assert result.test_name == 'Mean Ergodicity'
        assert result.is_ergodic is True

    def test_growth_rate_ergodicity(self):
        """Test growth rate ergodicity (Ole Peters' example)."""
        from theory.ergodicity import ErgodicityAnalyzer

        np.random.seed(42)
        # +50% or -40% gamble
        returns = np.random.choice([0.5, -0.4], size=500)

        analyzer = ErgodicityAnalyzer()
        result = analyzer.test_growth_rate_ergodicity(returns)

        # This gamble is NOT ergodic (arithmetic mean > 0, geometric mean < 0)
        assert result.time_average < result.ensemble_estimate


class TestPhase11Symmetries:
    """Tests for symmetry analysis."""

    def test_scale_invariance(self):
        """Test scale invariance check."""
        from theory.symmetries import test_scale_invariance

        np.random.seed(42)
        # Fractal/self-similar data
        data = np.cumsum(np.random.randn(1000))

        result = test_scale_invariance(data)

        assert 'is_scale_invariant' in result


class TestPhase12Fetchers:
    """Tests for multi-domain data fetchers."""

    def test_noaa_fetcher_import(self):
        """Test that NOAAFetcher can be imported."""
        from data.fetchers import NOAAFetcher
        fetcher = NOAAFetcher()
        assert fetcher is not None
        assert fetcher.source_name == "NOAA"

    def test_nasa_fetcher_import(self):
        """Test that NASAFetcher can be imported."""
        from data.fetchers import NASAFetcher
        fetcher = NASAFetcher()
        assert fetcher is not None
        assert fetcher.source_name == "NASA_GISS"

    def test_cdc_fetcher_import(self):
        """Test that CDCFetcher can be imported."""
        from data.fetchers import CDCFetcher
        fetcher = CDCFetcher()
        assert fetcher is not None
        assert fetcher.source_name == "CDC"

    def test_usgs_fetcher_import(self):
        """Test that USGSFetcher can be imported."""
        from data.fetchers import USGSFetcher
        fetcher = USGSFetcher()
        assert fetcher is not None
        assert fetcher.source_name == "USGS"

    def test_google_trends_fetcher_import(self):
        """Test that GoogleTrendsFetcher can be imported."""
        from data.fetchers import GoogleTrendsFetcher
        fetcher = GoogleTrendsFetcher()
        assert fetcher is not None
        assert fetcher.source_name == "GoogleTrends"

    def test_owid_fetcher_import(self):
        """Test that OWIDFetcher can be imported."""
        from data.fetchers import OWIDFetcher
        fetcher = OWIDFetcher()
        assert fetcher is not None
        assert fetcher.source_name == "OWID"

    def test_fetcher_synthetic_data(self):
        """Test that fetchers can generate synthetic data."""
        from data.fetchers import NOAAFetcher

        fetcher = NOAAFetcher()
        result = fetcher.fetch('enso', '2020-01-01', '2023-12-31')

        assert result.success is True
        assert result.data is not None
        assert len(result.data) > 0
        assert 'date' in result.data.columns
        assert 'value' in result.data.columns


class TestPhase12CrossDomain:
    """Tests for cross-domain coherence analysis."""

    def test_cross_domain_analyzer_import(self):
        """Test that CrossDomainAnalyzer can be imported."""
        from validation.cross_domain import CrossDomainAnalyzer, Domain
        analyzer = CrossDomainAnalyzer()
        assert analyzer is not None

    def test_add_series(self):
        """Test adding series to analyzer."""
        from validation.cross_domain import CrossDomainAnalyzer, Domain

        analyzer = CrossDomainAnalyzer()

        dates = pd.date_range('2020-01-01', periods=100, freq='W')
        data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100)
        })

        analyzer.add_series('test', Domain.CLIMATE, data, 'synthetic')

        assert 'test' in analyzer.series
        assert analyzer.series['test'].domain == Domain.CLIMATE

    def test_compute_coherence(self):
        """Test coherence computation."""
        from validation.cross_domain import CrossDomainAnalyzer, Domain

        np.random.seed(42)
        analyzer = CrossDomainAnalyzer(min_overlap=50)

        # Create correlated series
        dates = pd.date_range('2020-01-01', periods=200, freq='W')
        driver = np.sin(np.linspace(0, 10*np.pi, 200))

        data1 = pd.DataFrame({
            'date': dates,
            'value': driver + np.random.randn(200) * 0.3
        })

        data2 = pd.DataFrame({
            'date': dates,
            'value': driver + np.random.randn(200) * 0.3
        })

        analyzer.add_series('series1', Domain.CLIMATE, data1, 'synthetic')
        analyzer.add_series('series2', Domain.FINANCE, data2, 'synthetic')

        result = analyzer.compute_coherence('series1', 'series2', test_significance=False)

        assert result is not None
        assert result.coherence > 0.3  # Should have some coherence
        assert result.n_observations == 200


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
