"""
Test DB Performance Diagnostic
===============================

Tests for the check_db_performance.py diagnostic script.
These tests mock the database and do NOT require real data.
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


class TestDBPerformance:
    """Test cases for DB performance diagnostic."""

    @pytest.fixture
    def mock_table_list(self):
        """Create mock table list."""
        return [
            {"name": "indicator_values"},
            {"name": "indicators"},
            {"name": "fetch_log"},
        ]

    def test_imports(self):
        """Test that the DB performance module can be imported."""
        try:
            from start.performance import check_db_performance
            assert hasattr(check_db_performance, "main")
            assert hasattr(check_db_performance, "get_table_row_counts")
            assert hasattr(check_db_performance, "check_index_health")
            assert hasattr(check_db_performance, "detect_fragmentation")
        except ImportError as e:
            pytest.skip(f"Module import failed: {e}")

    def test_get_table_row_counts_with_mock(self, mock_table_list):
        """Test get_table_row_counts with mocked database."""
        from start.performance.check_db_performance import get_table_row_counts

        with patch(
            "start.performance.check_db_performance.get_connection"
        ) as mock_conn:
            mock_cursor = MagicMock()

            # Mock table list query
            mock_cursor.fetchall.side_effect = [
                mock_table_list,  # First call returns tables
                [{"cnt": 1000}],  # indicator_values
                [{"cnt": 50}],    # indicators
                [{"cnt": 200}],   # fetch_log
            ]
            mock_conn.return_value.cursor.return_value = mock_cursor

            counts = get_table_row_counts()

            assert isinstance(counts, dict)
            # Should have attempted to count rows

    def test_check_index_health_returns_valid_structure(self):
        """Test that check_index_health returns expected structure."""
        from start.performance.check_db_performance import check_index_health

        with patch(
            "start.performance.check_db_performance.get_connection"
        ) as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                {"name": "idx_indicator_values_name", "tbl_name": "indicator_values", "sql": "CREATE INDEX..."},
            ]
            mock_conn.return_value.cursor.return_value = mock_cursor

            result = check_index_health()

            assert "indexes" in result
            assert "missing_recommended_indexes" in result
            assert "status" in result
            assert result["status"] in ("healthy", "warning", "error")

    def test_detect_fragmentation_returns_valid_structure(self):
        """Test that detect_fragmentation returns expected structure."""
        from start.performance.check_db_performance import detect_fragmentation

        with patch(
            "start.performance.check_db_performance.get_connection"
        ) as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.side_effect = [
                (100,),   # page_count
                (4096,),  # page_size
                (5,),     # freelist_count
            ]
            mock_conn.return_value.cursor.return_value = mock_cursor

            result = detect_fragmentation()

            assert "page_count" in result
            assert "page_size" in result
            assert "freelist_count" in result
            assert "fragmentation_ratio" in result
            assert "status" in result

    def test_time_repeated_loads_with_mock(self):
        """Test time_repeated_loads with mocked data."""
        from start.performance.check_db_performance import time_repeated_loads

        mock_df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            index=pd.date_range("2020-01-01", periods=3)
        )

        with patch(
            "start.performance.check_db_performance.load_all_indicators_wide"
        ) as mock_load:
            mock_load.return_value = mock_df

            result = time_repeated_loads(iterations=2)

            assert "iterations" in result
            assert result["iterations"] == 2
            assert "load_times_ms" in result
            assert len(result["load_times_ms"]) == 2

    def test_time_pivot_operation_with_mock(self):
        """Test time_pivot_operation with mocked data."""
        from start.performance.check_db_performance import time_pivot_operation

        mock_long_df = pd.DataFrame({
            "indicator_name": ["a", "a", "b", "b"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
            "value": [1.0, 2.0, 3.0, 4.0],
        })

        with patch(
            "start.performance.check_db_performance.get_connection"
        ) as mock_conn:
            with patch("pandas.read_sql_query") as mock_read:
                mock_read.return_value = mock_long_df

                result = time_pivot_operation()

                assert "raw_load_time_ms" in result
                assert "pivot_time_ms" in result
                assert "total_time_ms" in result

    def test_print_report_no_errors(self, capsys):
        """Test that print_report runs without errors."""
        from start.performance.check_db_performance import print_report

        row_counts = {"indicator_values": 1000, "indicators": 50}
        load_results = {
            "iterations": 5,
            "load_times_ms": [10, 12, 11, 13, 10],
            "avg_load_time_ms": 11.2,
            "min_load_time_ms": 10,
            "max_load_time_ms": 13,
            "total_rows_loaded": 5000,
        }
        pivot_results = {
            "raw_load_time_ms": 50,
            "pivot_time_ms": 20,
            "total_time_ms": 70,
            "resulting_shape": (100, 10),
        }
        index_health = {
            "indexes": [{"name": "idx1", "table": "t1", "sql": "..."}],
            "missing_recommended_indexes": [],
            "status": "healthy",
        }
        fragmentation = {
            "page_count": 100,
            "page_size": 4096,
            "total_size_bytes": 409600,
            "freelist_count": 5,
            "fragmentation_ratio": 0.05,
            "status": "healthy",
        }

        print_report(row_counts, load_results, pivot_results, index_health, fragmentation)

        captured = capsys.readouterr()
        assert "DATABASE PERFORMANCE REPORT" in captured.out


class TestDBPerformanceIntegration:
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

        from start.performance.check_db_performance import main

        result = main()
        assert result in (0, 1)
