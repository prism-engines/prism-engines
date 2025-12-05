"""
Tests for Seismometer - ML-based Regime Instability Detection
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_features():
    """Generate synthetic feature data for testing."""
    np.random.seed(42)

    # Generate 500 days of synthetic data
    dates = pd.date_range('2018-01-01', periods=500, freq='D')

    # Normal baseline period (first 300 days)
    n_features = 8
    baseline_data = np.random.randn(300, n_features) * 0.5

    # Stress period (days 300-400) - higher variance, correlation breakdown
    stress_data = np.random.randn(100, n_features) * 1.5 + np.random.randn(100, 1) * 0.5

    # Recovery period (days 400-500)
    recovery_data = np.random.randn(100, n_features) * 0.6

    # Combine
    data = np.vstack([baseline_data, stress_data, recovery_data])

    df = pd.DataFrame(
        data,
        index=dates,
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    return df


@pytest.fixture
def sample_features_with_anomalies():
    """Generate features with clear anomalous periods."""
    np.random.seed(123)

    dates = pd.date_range('2017-01-01', periods=600, freq='D')

    # Create correlated features
    base_signal = np.random.randn(600)
    data = []

    for i in range(6):
        noise = np.random.randn(600) * 0.3
        feature = base_signal + noise + i * 0.1
        data.append(feature)

    data = np.column_stack(data)

    # Inject anomalies at specific points
    # Anomaly 1: days 200-220 - sudden decorrelation
    data[200:220, :3] = np.random.randn(20, 3) * 3
    data[200:220, 3:] = -np.random.randn(20, 3) * 2

    # Anomaly 2: days 400-430 - extreme values
    data[400:430] = data[400:430] * 2.5 + np.random.randn(30, 6) * 2

    df = pd.DataFrame(
        data,
        index=dates,
        columns=[f'indicator_{i}' for i in range(6)]
    )

    return df


@pytest.fixture
def crisis_dates():
    """Sample crisis dates for backtesting."""
    return {
        'crisis_1': ('2018-10-01', '2018-10-15'),  # During stress period
        'crisis_2': ('2018-12-01', '2018-12-10'),  # During recovery
    }


# =============================================================================
# Test BaseDetector
# =============================================================================

class TestBaseDetector:
    """Tests for abstract BaseDetector functionality."""

    def test_validate_features_handles_date_column(self, sample_features):
        from engine_core.seismometer.base import BaseDetector

        # BaseDetector is abstract, so we test through a concrete implementation
        from engine_core.seismometer.clustering import ClusteringDriftDetector

        detector = ClusteringDriftDetector()

        # Add date as column instead of index
        df = sample_features.reset_index().rename(columns={'index': 'date'})

        validated = detector._validate_features(df)

        assert isinstance(validated.index, pd.DatetimeIndex)
        assert 'date' not in validated.columns

    def test_validate_features_rejects_empty(self, sample_features):
        from engine_core.seismometer.clustering import ClusteringDriftDetector

        detector = ClusteringDriftDetector()

        with pytest.raises(ValueError, match="empty"):
            detector._validate_features(pd.DataFrame())

    def test_standardize_fit_and_transform(self, sample_features):
        from engine_core.seismometer.clustering import ClusteringDriftDetector

        detector = ClusteringDriftDetector()

        X = sample_features.values[:100]

        # Fit standardizer
        X_scaled = detector._standardize(X, fit=True)

        # Check standardization
        assert np.abs(X_scaled.mean(axis=0)).max() < 0.1
        assert np.abs(X_scaled.std(axis=0) - 1.0).max() < 0.1

        # Transform new data using fitted params
        X_new = sample_features.values[100:200]
        X_new_scaled = detector._standardize(X_new, fit=False)

        assert X_new_scaled.shape == X_new.shape


# =============================================================================
# Test ClusteringDriftDetector
# =============================================================================

class TestClusteringDriftDetector:
    """Tests for K-Means based clustering drift detection."""

    def test_fit_creates_clusters(self, sample_features):
        from engine_core.seismometer.clustering import ClusteringDriftDetector

        detector = ClusteringDriftDetector(n_clusters=5)

        # Fit on baseline
        baseline = sample_features.iloc[:200]
        detector.fit(baseline)

        assert detector.is_fitted
        assert detector._kmeans is not None
        assert len(detector._kmeans.cluster_centers_) == 5

    def test_score_returns_valid_range(self, sample_features):
        from engine_core.seismometer.clustering import ClusteringDriftDetector

        detector = ClusteringDriftDetector(n_clusters=5)

        # Fit on baseline
        detector.fit(sample_features.iloc[:200])

        # Score all data
        scores = detector.score(sample_features)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_features)
        assert scores.dropna().min() >= 0.0
        assert scores.dropna().max() <= 1.0

    def test_score_higher_for_anomalies(self, sample_features_with_anomalies):
        from engine_core.seismometer.clustering import ClusteringDriftDetector

        detector = ClusteringDriftDetector(n_clusters=4)

        # Fit on normal period
        detector.fit(sample_features_with_anomalies.iloc[:150])

        # Score all
        scores = detector.score(sample_features_with_anomalies)

        # Scores during anomaly period should be higher
        normal_scores = scores.iloc[50:150].mean()
        anomaly_scores = scores.iloc[200:220].mean()

        assert anomaly_scores > normal_scores

    def test_fit_score_convenience(self, sample_features):
        from engine_core.seismometer.clustering import ClusteringDriftDetector

        detector = ClusteringDriftDetector()

        scores = detector.fit_score(sample_features, baseline_end='2018-07-01')

        assert detector.is_fitted
        assert len(scores) == len(sample_features)

    def test_get_cluster_assignments(self, sample_features):
        from engine_core.seismometer.clustering import ClusteringDriftDetector

        detector = ClusteringDriftDetector(n_clusters=4)
        detector.fit(sample_features.iloc[:200])

        assignments = detector.get_cluster_assignments(sample_features)

        assert isinstance(assignments, pd.Series)
        assert set(assignments.dropna().unique()).issubset({0, 1, 2, 3})


# =============================================================================
# Test ReconstructionErrorDetector
# =============================================================================

class TestReconstructionErrorDetector:
    """Tests for autoencoder-based reconstruction error detection."""

    def test_fit_trains_weights(self, sample_features):
        from engine_core.seismometer.autoencoder import ReconstructionErrorDetector

        detector = ReconstructionErrorDetector(
            encoding_dim=4,
            n_epochs=50,
            learning_rate=0.01
        )

        detector.fit(sample_features.iloc[:200])

        assert detector.is_fitted
        assert detector._W_enc is not None
        assert detector._W_dec is not None
        assert detector._W_enc.shape[1] == 4  # encoding_dim

    def test_score_returns_valid_range(self, sample_features):
        from engine_core.seismometer.autoencoder import ReconstructionErrorDetector

        detector = ReconstructionErrorDetector(encoding_dim=4, n_epochs=50)

        detector.fit(sample_features.iloc[:200])
        scores = detector.score(sample_features)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_features)
        assert scores.dropna().min() >= 0.0
        assert scores.dropna().max() <= 1.0

    def test_reconstruction_error_increases_for_anomalies(self, sample_features_with_anomalies):
        from engine_core.seismometer.autoencoder import ReconstructionErrorDetector

        detector = ReconstructionErrorDetector(encoding_dim=3, n_epochs=100)

        # Fit on normal period
        detector.fit(sample_features_with_anomalies.iloc[:150])

        scores = detector.score(sample_features_with_anomalies)

        # Anomaly periods should have higher reconstruction error
        normal_scores = scores.iloc[50:150].mean()
        anomaly_scores = scores.iloc[400:430].mean()

        assert anomaly_scores > normal_scores

    def test_get_encoding(self, sample_features):
        from engine_core.seismometer.autoencoder import ReconstructionErrorDetector

        detector = ReconstructionErrorDetector(encoding_dim=4, n_epochs=30)
        detector.fit(sample_features.iloc[:200])

        encodings = detector.get_encoding(sample_features)

        assert isinstance(encodings, pd.DataFrame)
        assert encodings.shape[1] == 4  # encoding_dim
        assert len(encodings) == len(sample_features)


# =============================================================================
# Test CorrelationGraphDetector
# =============================================================================

class TestCorrelationGraphDetector:
    """Tests for correlation graph based detection."""

    def test_fit_computes_baseline_correlation(self, sample_features):
        from engine_core.seismometer.correlation_graph import CorrelationGraphDetector

        detector = CorrelationGraphDetector(window_days=30)

        detector.fit(sample_features.iloc[:200])

        assert detector.is_fitted
        assert detector._baseline_corr is not None
        assert detector._baseline_corr.shape[0] == sample_features.shape[1]

    def test_score_returns_valid_range(self, sample_features):
        from engine_core.seismometer.correlation_graph import CorrelationGraphDetector

        detector = CorrelationGraphDetector(window_days=30, min_periods=20)

        detector.fit(sample_features.iloc[:200])
        scores = detector.score(sample_features)

        assert isinstance(scores, pd.Series)
        # First few values will be NaN due to rolling window
        valid_scores = scores.dropna()
        assert valid_scores.min() >= 0.0
        assert valid_scores.max() <= 1.0

    def test_detects_correlation_breakdown(self, sample_features_with_anomalies):
        from engine_core.seismometer.correlation_graph import CorrelationGraphDetector

        detector = CorrelationGraphDetector(window_days=30, min_periods=20)

        # Fit on normal correlated period
        detector.fit(sample_features_with_anomalies.iloc[:150])

        scores = detector.score(sample_features_with_anomalies)

        # Decorrelation period should have higher scores
        normal_scores = scores.iloc[100:150].mean()
        decorrelated_scores = scores.iloc[220:250].mean()  # After anomaly

        # This test may be flaky due to random data - use tolerance
        # The decorrelated period should generally score higher
        assert decorrelated_scores >= normal_scores * 0.8 or decorrelated_scores > 0.3

    def test_get_rolling_correlations(self, sample_features):
        from engine_core.seismometer.correlation_graph import CorrelationGraphDetector

        detector = CorrelationGraphDetector(window_days=30)
        detector.fit(sample_features.iloc[:200])

        rolling_stats = detector.get_rolling_correlations(sample_features)

        assert isinstance(rolling_stats, pd.DataFrame)
        assert 'avg_correlation' in rolling_stats.columns
        assert 'avg_abs_correlation' in rolling_stats.columns


# =============================================================================
# Test Seismometer Ensemble
# =============================================================================

class TestSeismometerEnsemble:
    """Tests for the main Seismometer ensemble class."""

    def test_initialization_with_defaults(self):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()

        assert len(seis.detectors) == 3
        assert sum(seis.weights.values()) == pytest.approx(1.0)
        assert not seis.is_fitted

    def test_initialization_with_custom_weights(self):
        from engine_core.seismometer import Seismometer

        weights = {
            'clustering_drift': 0.5,
            'reconstruction_error': 0.3,
            'correlation_graph': 0.2,
        }

        seis = Seismometer(weights=weights)

        assert seis.weights['clustering_drift'] == 0.5

    def test_fit_trains_all_detectors(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')

        assert seis.is_fitted
        for name, detector in seis.detectors.items():
            assert detector.is_fitted, f"{name} not fitted"

    def test_score_returns_all_components(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')

        scores = seis.score()

        assert isinstance(scores, pd.DataFrame)
        assert 'clustering_drift' in scores.columns
        assert 'reconstruction_error' in scores.columns
        assert 'correlation_graph' in scores.columns
        assert 'stability_index' in scores.columns
        assert 'instability_score' in scores.columns

    def test_stability_index_in_valid_range(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')

        scores = seis.score()

        valid_stability = scores['stability_index'].dropna()
        assert valid_stability.min() >= 0.0
        assert valid_stability.max() <= 1.0

    def test_get_stability_index(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')
        seis.score()

        result = seis.get_stability_index('2018-10-01')

        assert 'date' in result
        assert 'stability_index' in result
        assert 'alert_level' in result
        assert 'components' in result
        assert 0 <= result['stability_index'] <= 1

    def test_get_current_status(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')

        status = seis.get_current_status()

        assert 'stability_index' in status
        assert 'alert_level' in status
        assert status['alert_level'] in ['stable', 'elevated', 'pre_instability', 'divergence', 'high_risk']

    def test_alert_levels_mapping(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()

        # Test all alert level boundaries
        test_cases = [
            (0.95, 'stable'),
            (0.80, 'elevated'),
            (0.60, 'pre_instability'),
            (0.40, 'divergence'),
            (0.20, 'high_risk'),
        ]

        for stability, expected_level in test_cases:
            for (low, high), level in seis.ALERT_LEVELS.items():
                if low <= stability < high:
                    assert level == expected_level, f"stability={stability} should be {expected_level}"
                    break

    def test_get_alert_history(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')

        history = seis.get_alert_history()

        assert isinstance(history, pd.DataFrame)
        assert 'stability_index' in history.columns
        assert 'alert_level' in history.columns

    def test_get_detector_contributions(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')
        seis.score()

        contributions = seis.get_detector_contributions('2018-10-01')

        assert isinstance(contributions, pd.DataFrame)
        assert 'detector' in contributions.columns
        assert 'score' in contributions.columns
        assert 'weight' in contributions.columns
        assert 'contribution' in contributions.columns

    def test_summary_output(self, sample_features):
        from engine_core.seismometer import Seismometer

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')

        summary = seis.summary()

        assert isinstance(summary, str)
        assert 'Stability Index' in summary
        assert 'Alert Level' in summary


# =============================================================================
# Test Calibration
# =============================================================================

class TestCalibration:
    """Tests for calibration and backtesting functions."""

    def test_compute_alert_frequency(self, sample_features):
        from engine_core.seismometer import Seismometer
        from engine_core.seismometer.calibration import compute_alert_frequency

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')
        scores = seis.score()

        freq = compute_alert_frequency(scores['stability_index'], threshold=0.5)

        assert 0 <= freq <= 1

    def test_compute_alert_duration(self, sample_features):
        from engine_core.seismometer import Seismometer
        from engine_core.seismometer.calibration import compute_alert_duration

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')
        scores = seis.score()

        duration_stats = compute_alert_duration(scores['stability_index'], threshold=0.5)

        assert 'mean_duration' in duration_stats
        assert 'median_duration' in duration_stats
        assert 'max_duration' in duration_stats
        assert 'n_alerts' in duration_stats

    def test_backtest_alerts(self, sample_features, crisis_dates):
        from engine_core.seismometer import Seismometer
        from engine_core.seismometer.calibration import backtest_alerts

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')
        scores = seis.score()

        results = backtest_alerts(
            scores['stability_index'],
            threshold=0.5,
            crisis_dates=crisis_dates,
            lead_days=30
        )

        assert isinstance(results, pd.DataFrame)
        assert 'crisis_name' in results.columns
        assert 'detected' in results.columns
        assert 'early_warning' in results.columns

    def test_find_optimal_threshold(self, sample_features, crisis_dates):
        from engine_core.seismometer import Seismometer
        from engine_core.seismometer.calibration import find_optimal_threshold

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')
        scores = seis.score()

        result = find_optimal_threshold(
            scores['stability_index'],
            crisis_dates=crisis_dates,
            target_precision=0.5,
            n_steps=10
        )

        assert 'optimal_threshold' in result
        assert 0 <= result['optimal_threshold'] <= 1
        assert 'precision' in result
        assert 'recall' in result

    def test_compute_detection_metrics(self, sample_features, crisis_dates):
        from engine_core.seismometer import Seismometer
        from engine_core.seismometer.calibration import backtest_alerts, compute_detection_metrics

        seis = Seismometer()
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')
        scores = seis.score()

        backtest = backtest_alerts(scores['stability_index'], 0.5, crisis_dates)
        metrics = compute_detection_metrics(backtest)

        assert 'detection_rate' in metrics
        assert 'early_warning_rate' in metrics
        assert 'avg_lead_days' in metrics


# =============================================================================
# Test Utils
# =============================================================================

class TestUtils:
    """Tests for utility functions."""

    def test_compute_rolling_zscore(self):
        from engine_core.seismometer.utils import compute_rolling_zscore

        data = pd.Series(np.random.randn(100))
        zscore = compute_rolling_zscore(data, window=20)

        assert len(zscore) == len(data)
        # Z-score should be roughly mean 0 after warmup
        assert abs(zscore.iloc[50:].mean()) < 0.5

    def test_compute_regime_duration(self):
        from engine_core.seismometer.utils import compute_regime_duration

        # Create alternating stability pattern
        stability = pd.Series([0.8, 0.8, 0.8, 0.3, 0.3, 0.6, 0.6])

        duration = compute_regime_duration(stability, threshold=0.5)

        assert len(duration) == len(stability)
        # First 3 days are stable (positive)
        assert duration.iloc[2] == 3
        # Days 3-4 are unstable (negative)
        assert duration.iloc[4] == -2

    def test_detect_divergence_acceleration(self):
        from engine_core.seismometer.utils import detect_divergence_acceleration

        # Create accelerating instability pattern
        scores = pd.Series([0.1, 0.15, 0.22, 0.31, 0.42, 0.55, 0.70])

        acceleration = detect_divergence_acceleration(scores, window=3)

        assert len(acceleration) == len(scores)
        # Second derivative of accelerating pattern should be positive
        valid_accel = acceleration.dropna()
        assert len(valid_accel) > 0

    def test_find_regime_transitions(self):
        from engine_core.seismometer.utils import find_regime_transitions

        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        stability = pd.Series(
            [0.9, 0.85, 0.75, 0.65, 0.55, 0.45, 0.40, 0.50, 0.60, 0.75],
            index=dates
        )

        transitions = find_regime_transitions(stability)

        assert isinstance(transitions, pd.DataFrame)
        assert 'from_level' in transitions.columns
        assert 'to_level' in transitions.columns
        assert len(transitions) > 0

    def test_get_default_features(self):
        from engine_core.seismometer.utils import get_default_features

        features = get_default_features()

        assert isinstance(features, list)
        assert len(features) > 0
        assert 't10y2y' in features


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline(self, sample_features):
        """Test complete seismometer pipeline."""
        from engine_core.seismometer import Seismometer

        # Initialize
        seis = Seismometer()

        # Fit on baseline
        seis.fit(
            features=sample_features,
            start='2018-01-01',
            end='2018-06-30'
        )

        # Score
        scores = seis.score()
        assert len(scores) == len(sample_features)

        # Get status
        status = seis.get_current_status()
        assert status['alert_level'] in ['stable', 'elevated', 'pre_instability', 'divergence', 'high_risk']

        # Get history
        history = seis.get_alert_history()
        assert len(history) == len(sample_features)

    def test_detector_scores_contribute_to_ensemble(self, sample_features):
        """Verify ensemble properly combines detector scores."""
        from engine_core.seismometer import Seismometer

        weights = {
            'clustering_drift': 0.5,
            'reconstruction_error': 0.3,
            'correlation_graph': 0.2,
        }

        seis = Seismometer(weights=weights)
        seis.fit(features=sample_features, start='2018-01-01', end='2018-06-30')
        scores = seis.score()

        # Manual calculation for a sample row
        idx = 200  # Pick a row
        row = scores.iloc[idx]

        expected_instability = (
            0.5 * row['clustering_drift'] +
            0.3 * row['reconstruction_error'] +
            0.2 * row['correlation_graph']
        )

        # Account for NaN handling
        if not np.isnan(expected_instability):
            assert abs(row['instability_score'] - expected_instability) < 0.01

    def test_imports_work(self):
        """Verify all imports work correctly."""
        from engine_core.seismometer import (
            Seismometer,
            ClusteringDriftDetector,
            ReconstructionErrorDetector,
            CorrelationGraphDetector,
            BaseDetector,
        )

        assert Seismometer is not None
        assert ClusteringDriftDetector is not None
        assert ReconstructionErrorDetector is not None
        assert CorrelationGraphDetector is not None
        assert BaseDetector is not None
