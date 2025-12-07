"""
Test Fetch Performance Diagnostic
==================================

Tests for the check_fetch_performance.py diagnostic script.
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


class TestFetchPerformance:
    """Test cases for fetch performance diagnostic."""

    @pytest.fixture
    def mock_panel_data(self):
        """Create mock panel data for testing."""
        dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
        data = {
            "indicator_1": np.random.randn(252),
            "indicator_2": np.random.randn(252),
            "indicator_3": np.random.randn(252),
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def mock_indicator_rows(self):
        """Create mock indicator metadata rows."""
        return [
            {"name": "gdp_growth", "source": "fred"},
            {"name": "unemployment_rate", "source": "fred"},
            {"name": "sp500", "source": "tiingo"},
            {"name": "vix", "source": "tiingo"},
        ]

    def test_imports(self):
        """Test that the fetch performance module can be imported."""
        try:
            from start.performance import check_fetch_performance
            assert hasattr(check_fetch_performance, "main")
            assert hasattr(check_fetch_performance, "time_fetch")
            assert hasattr(check_fetch_performance, "print_report")
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")

    def test_time_fetch_with_mock_data(self, mock_panel_data):
        """Test time_fetch function with mocked data."""
        from start.performance.check_fetch_performance import time_fetch

        with patch(
            "start.performance.check_fetch_performance.load_all_indicators_wide"
        ) as mock_load:
            mock_load.return_value = mock_panel_data

            results = time_fetch(["indicator_1", "indicator_2"], "test_source")

            assert "source" in results
            assert results["source"] == "test_source"
            assert "fetch_times_ms" in results
            assert "stale_indicators" in results
            assert "missing_indicators" in results

    def test_time_fetch_handles_missing_indicator(self, mock_panel_data):
        """Test that time_fetch handles missing indicators gracefully."""
        from start.performance.check_fetch_performance import time_fetch

        # Return empty DataFrame for missing indicator
        def mock_load_side_effect(indicators=None, **kwargs):
            if indicators and "missing_indicator" in indicators:
                return pd.DataFrame()
            return mock_panel_data

        with patch(
            "start.performance.check_fetch_performance.load_all_indicators_wide",
            side_effect=mock_load_side_effect
        ):
            results = time_fetch(["missing_indicator"], "test_source")

            assert "missing_indicator" in results["missing_indicators"]

    def test_get_available_indicators_by_source(self, mock_indicator_rows):
        """Test grouping indicators by source."""
        from start.performance.check_fetch_performance import (
            get_available_indicators_by_source
        )

        with patch(
            "start.performance.check_fetch_performance.get_connection"
        ) as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = mock_indicator_rows
            mock_conn.return_value.cursor.return_value = mock_cursor

            fred, tiingo = get_available_indicators_by_source()

            # Should have grouped correctly
            assert isinstance(fred, list)
            assert isinstance(tiingo, list)

    def test_staleness_detection(self, mock_panel_data):
        """Test that stale data is detected correctly."""
        from start.performance.check_fetch_performance import time_fetch

        # Create data that's stale (more than 7 days old)
        old_dates = pd.date_range(
            start="2020-01-01",
            periods=252,
            freq="D"
        )
        stale_data = mock_panel_data.copy()
        stale_data.index = old_dates

        with patch(
            "start.performance.check_fetch_performance.load_all_indicators_wide"
        ) as mock_load:
            mock_load.return_value = stale_data

            results = time_fetch(["indicator_1"], "test_source")

            # Should detect staleness
            assert len(results["stale_indicators"]) > 0

    def test_print_report_no_errors(self, capsys):
        """Test that print_report runs without errors."""
        from start.performance.check_fetch_performance import print_report

        fred_results = {
            "source": "FRED",
            "indicators": ["gdp"],
            "fetch_times_ms": [10.5, 12.3],
            "latest_dates": {},
            "stale_indicators": [],
            "missing_indicators": [],
            "errors": [],
        }

        tiingo_results = {
            "source": "Tiingo",
            "indicators": ["sp500"],
            "fetch_times_ms": [8.1],
            "latest_dates": {},
            "stale_indicators": [],
            "missing_indicators": [],
            "errors": [],
        }

        # Should not raise any exceptions
        print_report(fred_results, tiingo_results)

        captured = capsys.readouterr()
        assert "FETCH PERFORMANCE REPORT" in captured.out


class TestFetchPerformanceIntegration:
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

        from start.performance.check_fetch_performance import main

        # Should return 0 or 1, not raise exception
        result = main()
        assert result in (0, 1)
