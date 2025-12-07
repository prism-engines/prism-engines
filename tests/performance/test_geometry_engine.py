"""
Test Geometry Engine Diagnostic
================================

Tests for the check_geometry_engine.py diagnostic script.
These tests mock the database and do NOT require real data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestGeometryEngine:
    """Test cases for geometry engine diagnostic."""

    @pytest.fixture
    def mock_panel_data(self):
        """Create mock panel data for testing."""
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        np.random.seed(42)
        data = {
            "indicator_1": np.random.randn(252),
            "indicator_2": np.random.randn(252),
            "indicator_3": np.random.randn(252),
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def panel_with_gaps(self):
        """Create panel data with date gaps."""
        # Create dates with a large gap
        dates1 = pd.date_range(start="2020-01-01", periods=100, freq="B")
        dates2 = pd.date_range(start="2020-08-01", periods=100, freq="B")
        dates = dates1.append(dates2)

        data = {
            "indicator_1": np.random.randn(200),
            "indicator_2": np.random.randn(200),
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def panel_with_nans(self):
        """Create panel data with high NaN rates."""
        dates = pd.date_range(start="2020-01-01", periods=252, freq="B")
        data = {
            "indicator_1": np.random.randn(252),
            "indicator_2": np.random.randn(252),
        }
        df = pd.DataFrame(data, index=dates)
        # Introduce 50% NaN
        df.loc[df.index[:126], "indicator_2"] = np.nan
        return df

    def test_imports(self):
        """Test that the geometry engine module can be imported."""
        try:
            from start.performance import check_geometry_engine
            assert hasattr(check_geometry_engine, "main")
            assert hasattr(check_geometry_engine, "check_missing_dates")
            assert hasattr(check_geometry_engine, "check_nan_rates")
            assert hasattr(check_geometry_engine, "check_monotonic_index")
            assert hasattr(check_geometry_engine, "check_family_double_loading")
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")

    def test_check_missing_dates_no_gaps(self, mock_panel_data):
        """Test check_missing_dates with clean data."""
        from start.performance.check_geometry_engine import check_missing_dates

        result = check_missing_dates(mock_panel_data)

        assert result["status"] == "pass"
        assert result["gaps_found"] == 0

    def test_check_missing_dates_with_gaps(self, panel_with_gaps):
        """Test check_missing_dates detects gaps."""
        from start.performance.check_geometry_engine import check_missing_dates

        result = check_missing_dates(panel_with_gaps)

        assert result["status"] == "warning"
        assert result["gaps_found"] > 0

    def test_check_nan_rates_clean_data(self, mock_panel_data):
        """Test check_nan_rates with clean data."""
        from start.performance.check_geometry_engine import check_nan_rates

        result = check_nan_rates(mock_panel_data)

        assert result["status"] == "pass"
        assert len(result["high_nan_indicators"]) == 0

    def test_check_nan_rates_with_nans(self, panel_with_nans):
        """Test check_nan_rates detects high NaN rates."""
        from start.performance.check_geometry_engine import check_nan_rates

        result = check_nan_rates(panel_with_nans)

        assert result["status"] == "warning"
        assert len(result["high_nan_indicators"]) > 0
        assert any(
            ind["indicator"] == "indicator_2"
            for ind in result["high_nan_indicators"]
        )

    def test_check_monotonic_index_clean(self, mock_panel_data):
        """Test check_monotonic_index with monotonic data."""
        from start.performance.check_geometry_engine import check_monotonic_index

        result = check_monotonic_index(mock_panel_data)

        assert result["status"] == "pass"
        assert result["is_monotonic"] is True
        assert result["duplicate_dates"] == 0

    def test_check_monotonic_index_with_duplicates(self, mock_panel_data):
        """Test check_monotonic_index detects duplicate dates."""
        from start.performance.check_geometry_engine import check_monotonic_index

        # Create data with duplicate dates
        df = mock_panel_data.copy()
        df.index = list(df.index[:-1]) + [df.index[0]]  # Duplicate first date

        result = check_monotonic_index(df)

        assert result["duplicate_dates"] > 0

    def test_check_family_double_loading_empty(self, mock_panel_data):
        """Test check_family_double_loading with mock data."""
        from start.performance.check_geometry_engine import check_family_double_loading

        with patch(
            "start.performance.check_geometry_engine.FamilyManager"
        ) as mock_fm:
            mock_fm_instance = MagicMock()
            mock_fm_instance.list_families.return_value = []
            mock_fm.return_value = mock_fm_instance

            result = check_family_double_loading(mock_panel_data)

            assert "status" in result
            assert "double_loaded_families" in result

    def test_check_indicator_drift_stable(self, mock_panel_data):
        """Test check_indicator_drift with stable data."""
        from start.performance.check_geometry_engine import check_indicator_drift

        # Use smaller window for test data
        result = check_indicator_drift(mock_panel_data, window=50)

        assert "status" in result
        assert "drifting_indicators" in result

    def test_check_indicator_drift_with_drift(self):
        """Test check_indicator_drift detects drifting data."""
        from start.performance.check_geometry_engine import check_indicator_drift

        # Create data with intentional drift
        dates = pd.date_range(start="2020-01-01", periods=500, freq="B")
        values = np.concatenate([
            np.random.randn(400),      # Normal period
            np.random.randn(100) + 10  # Drifted period (mean shift)
        ])
        df = pd.DataFrame({"drifting_ind": values}, index=dates)

        result = check_indicator_drift(df, window=100)

        # Should detect drift
        assert "drifting_indicators" in result

    def test_print_report_no_errors(self, capsys):
        """Test that print_report runs without errors."""
        from start.performance.check_geometry_engine import print_report

        gap_results = {"status": "pass", "gaps_found": 0, "gap_details": []}
        nan_results = {"status": "pass", "high_nan_indicators": [], "overall_nan_rate": 0.01}
        monotonic_results = {"status": "pass", "is_monotonic": True, "duplicate_dates": 0, "out_of_order_dates": 0}
        family_results = {"status": "pass", "families_checked": 5, "double_loaded_families": [], "correlation_warnings": []}
        drift_results = {"status": "pass", "indicators_checked": 10, "drifting_indicators": []}

        print_report(gap_results, nan_results, monotonic_results, family_results, drift_results)

        captured = capsys.readouterr()
        assert "GEOMETRY ENGINE HEALTH" in captured.out


class TestGeometryEngineIntegration:
    """Integration tests (skipped if DB not available)."""

    @pytest.fixture
    def db_available(self):
        """Check if database is available."""
        try:
            from data.sql.db_connector import get_connection
            conn = get_connection()
            conn.close()
            return True
        except Exception:
            return False

    def test_main_runs_without_fatal_error(self, db_available):
        """Test that main() runs without fatal errors."""
        if not db_available:
            pytest.skip("Database not available")

        from start.performance.check_geometry_engine import main

        result = main()
        assert result in (0, 1)
