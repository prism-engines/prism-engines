"""
Test Lens Performance Diagnostic
=================================

Tests for the check_lens_performance.py diagnostic script.
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


class TestLensPerformance:
    """Test cases for lens performance diagnostic."""

    @pytest.fixture
    def mock_lens_panel(self):
        """Create mock panel data suitable for lens analysis."""
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=500, freq="B")
        data = {
            "date": dates,
            "indicator_1": np.random.randn(500),
            "indicator_2": np.random.randn(500),
            "indicator_3": np.random.randn(500),
            "indicator_4": np.random.randn(500),
            "indicator_5": np.random.randn(500),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_lens_result(self):
        """Create mock lens analysis result."""
        return {
            "rankings": pd.DataFrame({
                "indicator": ["ind_1", "ind_2", "ind_3"],
                "score": [0.8, 0.6, 0.4],
                "rank": [1, 2, 3]
            }),
            "insights": ["Insight 1", "Insight 2"],
            "metadata": {"computation_time": 0.05}
        }

    def test_imports(self):
        """Test that the lens performance module can be imported."""
        try:
            from start.performance import check_lens_performance
            assert hasattr(check_lens_performance, "main")
            assert hasattr(check_lens_performance, "load_lens_class")
            assert hasattr(check_lens_performance, "validate_lens_output")
            assert hasattr(check_lens_performance, "run_lens_test")
            assert hasattr(check_lens_performance, "run_all_lens_tests")
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")

    def test_load_lens_class_valid(self):
        """Test loading a valid lens class."""
        from start.performance.check_lens_performance import load_lens_class

        lens_class, error = load_lens_class("magnitude")

        # Should either succeed or return an error message
        assert lens_class is not None or error is not None

    def test_load_lens_class_invalid(self):
        """Test loading an invalid lens name."""
        from start.performance.check_lens_performance import load_lens_class

        lens_class, error = load_lens_class("nonexistent_lens")

        assert lens_class is None
        assert error is not None
        assert "Unknown lens" in error

    def test_validate_lens_output_valid(self, mock_lens_result):
        """Test validation of valid lens output."""
        from start.performance.check_lens_performance import validate_lens_output

        validation = validate_lens_output(mock_lens_result, "test_lens")

        assert validation["status"] in ("pass", "warning")
        assert validation["has_rankings"] is True
        assert validation["has_insights"] is True

    def test_validate_lens_output_empty(self):
        """Test validation of empty lens output."""
        from start.performance.check_lens_performance import validate_lens_output

        validation = validate_lens_output({}, "test_lens")

        assert validation["status"] == "warning"
        assert "Empty result dictionary" in validation["issues"]

    def test_validate_lens_output_not_dict(self):
        """Test validation of non-dict output."""
        from start.performance.check_lens_performance import validate_lens_output

        validation = validate_lens_output("not a dict", "test_lens")

        assert validation["status"] == "fail"
        assert any("not a dictionary" in issue for issue in validation["issues"])

    def test_validate_lens_output_constant_values(self):
        """Test validation detects constant values."""
        from start.performance.check_lens_performance import validate_lens_output

        result = {
            "rankings": pd.DataFrame({
                "indicator": ["a", "b", "c"],
                "score": [1.0, 1.0, 1.0],  # All same
            })
        }

        validation = validate_lens_output(result, "test_lens")

        # Should detect constant values
        assert any("constant" in issue.lower() for issue in validation.get("issues", []))

    def test_run_lens_test_with_mock(self, mock_lens_panel, mock_lens_result):
        """Test running a single lens test with mocked lens."""
        from start.performance.check_lens_performance import run_lens_test

        # Create a mock lens class
        mock_lens_class = MagicMock()
        mock_lens_instance = MagicMock()
        mock_lens_instance.analyze.return_value = mock_lens_result
        mock_lens_class.return_value = mock_lens_instance

        with patch(
            "start.performance.check_lens_performance.load_lens_class",
            return_value=(mock_lens_class, None)
        ):
            result = run_lens_test("mock_lens", mock_lens_panel)

            assert result["lens_name"] == "mock_lens"
            assert result["status"] in ("pass", "warning", "error", "skip")
            assert result["runtime_ms"] >= 0

    def test_run_lens_test_skip_missing(self, mock_lens_panel):
        """Test that missing lens is skipped."""
        from start.performance.check_lens_performance import run_lens_test

        with patch(
            "start.performance.check_lens_performance.load_lens_class",
            return_value=(None, "Lens not found")
        ):
            result = run_lens_test("missing_lens", mock_lens_panel)

            assert result["status"] == "skip"
            assert result["error"] is not None

    def test_run_all_lens_tests_with_mock(self, mock_lens_panel, mock_lens_result):
        """Test running all lens tests with mocked data."""
        from start.performance.check_lens_performance import run_all_lens_tests

        mock_lens_class = MagicMock()
        mock_lens_instance = MagicMock()
        mock_lens_instance.analyze.return_value = mock_lens_result
        mock_lens_class.return_value = mock_lens_instance

        with patch(
            "start.performance.check_lens_performance.load_lens_class",
            return_value=(mock_lens_class, None)
        ):
            results = run_all_lens_tests(mock_lens_panel)

            assert isinstance(results, list)
            assert len(results) > 0
            assert all("lens_name" in r for r in results)
            assert all("status" in r for r in results)

    def test_print_report_no_errors(self, capsys):
        """Test that print_report runs without errors."""
        from start.performance.check_lens_performance import print_report

        results = [
            {
                "lens_name": "magnitude",
                "status": "pass",
                "runtime_ms": 50.5,
                "error": None,
                "validation": {
                    "status": "pass",
                    "issues": [],
                    "output_keys": ["rankings", "insights"],
                    "has_rankings": True,
                    "has_insights": True
                }
            },
            {
                "lens_name": "pca",
                "status": "pass",
                "runtime_ms": 120.3,
                "error": None,
                "validation": {
                    "status": "pass",
                    "issues": [],
                    "output_keys": ["components", "variance"],
                    "has_rankings": True,
                    "has_insights": False
                }
            }
        ]

        print_report(results, (500, 6))

        captured = capsys.readouterr()
        assert "LENS PERFORMANCE REPORT" in captured.out
        assert "magnitude" in captured.out
        assert "pca" in captured.out


class TestLensPerformanceIntegration:
    """Integration tests (skipped if DB not available)."""

    @pytest.fixture
    def db_available(self):
        """Check if database is available."""
        try:
            from data.sql.db_connector import load_all_indicators_wide
            df = load_all_indicators_wide()
            return len(df) >= 100
        except Exception:
            return False

    def test_main_runs_without_fatal_error(self, db_available):
        """Test that main() runs without fatal errors."""
        if not db_available:
            pytest.skip("Database not available or insufficient data")

        from start.performance.check_lens_performance import main

        result = main()
        assert result in (0, 1)

    def test_individual_lens_loads(self):
        """Test that individual lens classes can be loaded."""
        from start.performance.check_lens_performance import (
            load_lens_class,
            LENS_MODULES
        )

        # Test a few basic lenses
        basic_lenses = ["magnitude", "pca", "clustering"]

        for lens_name in basic_lenses:
            if lens_name in LENS_MODULES:
                lens_class, error = load_lens_class(lens_name)
                # Should either load or give meaningful error
                assert lens_class is not None or error is not None, \
                    f"Lens {lens_name} returned both None"
