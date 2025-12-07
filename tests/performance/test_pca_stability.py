"""
Test PCA Stability Diagnostic
==============================

Tests for the check_pca_stability.py diagnostic script.
These tests mock heavy operations and do NOT require full DB.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPCAStability:
    """Test cases for PCA stability diagnostic."""

    @pytest.fixture
    def mock_panel_data(self):
        """Create mock panel data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2018-01-01", periods=1000, freq="B")
        # Create correlated data
        base = np.random.randn(1000)
        data = {
            f"indicator_{i}": base * (0.5 + np.random.rand()) + np.random.randn(1000) * 0.3
            for i in range(10)
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def stable_pca_data(self):
        """Create data with stable PCA structure."""
        np.random.seed(42)
        n = 1000
        # Create data with consistent factor structure
        factor1 = np.random.randn(n)
        factor2 = np.random.randn(n)

        data = {}
        for i in range(5):
            # First 3 indicators load on factor1
            if i < 3:
                data[f"ind_{i}"] = factor1 * 0.8 + np.random.randn(n) * 0.2
            # Last 2 load on factor2
            else:
                data[f"ind_{i}"] = factor2 * 0.8 + np.random.randn(n) * 0.2

        dates = pd.date_range(start="2018-01-01", periods=n, freq="B")
        return pd.DataFrame(data, index=dates)

    def test_imports(self):
        """Test that the PCA stability module can be imported."""
        try:
            from start.performance import check_pca_stability
            assert hasattr(check_pca_stability, "main")
            assert hasattr(check_pca_stability, "compute_eigenvector_angle")
            assert hasattr(check_pca_stability, "run_pca")
            assert hasattr(check_pca_stability, "compute_rolling_pca_drift")
            assert hasattr(check_pca_stability, "analyze_drift")
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")

    def test_compute_eigenvector_angle_parallel(self):
        """Test angle computation for parallel vectors."""
        from start.performance.check_pca_stability import compute_eigenvector_angle

        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])

        angle = compute_eigenvector_angle(v1, v2)

        # Parallel vectors should have 0 degree angle
        assert abs(angle) < 1e-10

    def test_compute_eigenvector_angle_perpendicular(self):
        """Test angle computation for perpendicular vectors."""
        from start.performance.check_pca_stability import compute_eigenvector_angle

        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])

        angle = compute_eigenvector_angle(v1, v2)

        # Perpendicular vectors should have 90 degree angle
        assert abs(angle - 90) < 1e-10

    def test_compute_eigenvector_angle_opposite(self):
        """Test angle computation for opposite vectors (sign flip)."""
        from start.performance.check_pca_stability import compute_eigenvector_angle

        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])

        angle = compute_eigenvector_angle(v1, v2)

        # Opposite vectors should still have 0 degree angle (we handle sign flips)
        assert abs(angle) < 1e-10

    def test_run_pca_basic(self, mock_panel_data):
        """Test basic PCA execution."""
        from start.performance.check_pca_stability import run_pca

        result = run_pca(mock_panel_data)

        assert result["error"] is None
        assert result["components"] is not None
        assert result["explained_variance_ratio"] is not None
        assert result["n_samples"] > 0
        assert result["n_features"] > 0

    def test_run_pca_insufficient_data(self):
        """Test PCA with insufficient data."""
        from start.performance.check_pca_stability import run_pca

        # Only 5 rows
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]},
            index=pd.date_range("2020-01-01", periods=5)
        )

        result = run_pca(df)

        assert result["error"] is not None

    def test_compute_rolling_pca_drift(self, stable_pca_data):
        """Test rolling PCA drift computation."""
        from start.performance.check_pca_stability import compute_rolling_pca_drift

        # Use smaller windows for test
        result = compute_rolling_pca_drift(
            stable_pca_data,
            window_days=200,
            step_days=50
        )

        assert "pc1_drifts" in result
        assert "pc2_drifts" in result
        assert "dates" in result
        assert result["windows_computed"] > 0

    def test_analyze_drift_stable(self, stable_pca_data):
        """Test drift analysis with stable data."""
        from start.performance.check_pca_stability import (
            compute_rolling_pca_drift,
            analyze_drift
        )

        drift_results = compute_rolling_pca_drift(
            stable_pca_data,
            window_days=200,
            step_days=50
        )

        analysis = analyze_drift(drift_results)

        assert "pc1_median_drift" in analysis
        assert "pc1_max_drift" in analysis
        assert "pc1_anomalies" in analysis
        assert "status" in analysis

    def test_analyze_drift_detects_anomaly(self):
        """Test that drift analysis detects large angle changes."""
        from start.performance.check_pca_stability import analyze_drift

        # Simulate drift results with large drift
        drift_results = {
            "pc1_drifts": [5, 10, 15, 45, 8, 12],  # 45 degree drift
            "pc2_drifts": [3, 5, 7, 10, 6, 4],
            "dates": pd.date_range("2020-01-01", periods=7)[1:],  # 6 dates
            "windows_computed": 6,
            "errors": []
        }

        analysis = analyze_drift(drift_results)

        # Should detect anomaly (45 > 30 threshold)
        assert len(analysis["pc1_anomalies"]) > 0 or analysis["pc1_max_drift"] > 30

    def test_print_report_no_errors(self, capsys):
        """Test that print_report runs without errors."""
        from start.performance.check_pca_stability import print_report

        drift_results = {
            "pc1_drifts": [5, 10, 15],
            "pc2_drifts": [3, 5, 7],
            "dates": pd.date_range("2020-01-01", periods=3),
            "windows_computed": 3,
            "errors": []
        }

        drift_analysis = {
            "pc1_median_drift": 10.0,
            "pc1_max_drift": 15.0,
            "pc1_anomalies": [],
            "pc2_median_drift": 5.0,
            "pc2_max_drift": 7.0,
            "pc2_anomalies": [],
            "status": "pass"
        }

        print_report(drift_results, drift_analysis, (1000, 10))

        captured = capsys.readouterr()
        assert "PCA STABILITY REPORT" in captured.out


class TestPCAStabilityIntegration:
    """Integration tests (skipped if DB not available or insufficient data)."""

    @pytest.fixture
    def db_available(self):
        """Check if database is available with sufficient data."""
        try:
            from data.sql.db_connector import load_all_indicators_wide
            df = load_all_indicators_wide()
            # Need at least 3 years of data
            return len(df) >= 756 + 63
        except Exception:
            return False

    def test_main_runs_without_fatal_error(self, db_available):
        """Test that main() runs without fatal errors."""
        if not db_available:
            pytest.skip("Database not available or insufficient data")

        from start.performance.check_pca_stability import main

        result = main()
        assert result in (0, 1)
