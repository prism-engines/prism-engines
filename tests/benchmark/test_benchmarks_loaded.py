"""
Benchmark Loading Validation Tests
==================================

Tests to verify benchmark datasets are correctly loaded into the PRISM database.

Required validations:
  - Indicators created for all columns
  - Timeseries row counts match CSV row counts
  - system == "benchmark"
  - fred_code starts with "BENCH_"
  - No NULL indicator_name

Run these tests after loading benchmarks:
    python prism_run.py --load-benchmarks
    pytest tests/benchmark/test_benchmarks_loaded.py -v
"""

import pytest
from pathlib import Path

import pandas as pd

from data.sql.db_connector import get_connection
from data.sql.load_benchmarks import (
    BENCHMARK_FILES,
    load_all_benchmarks,
    clear_benchmarks,
    _get_benchmark_dir,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def benchmark_dir():
    """Return the benchmark directory path."""
    return _get_benchmark_dir()


@pytest.fixture(scope="module")
def loaded_benchmarks(benchmark_dir):
    """
    Ensure benchmarks are loaded before tests run.

    This fixture loads benchmarks once for the test module.
    """
    # Check if benchmarks exist
    if not benchmark_dir.exists():
        pytest.skip(f"Benchmark directory not found: {benchmark_dir}")

    # Load benchmarks (will update existing if already loaded)
    stats = load_all_benchmarks(verbose=False)
    return stats


@pytest.fixture
def db_connection():
    """Provide a database connection for tests."""
    conn = get_connection()
    yield conn
    conn.close()


# =============================================================================
# TEST: INDICATORS CREATED FOR ALL COLUMNS
# =============================================================================

class TestIndicatorsCreated:
    """Test that indicators are created for all benchmark columns."""

    def test_all_benchmark_files_have_indicators(self, loaded_benchmarks, benchmark_dir, db_connection):
        """Verify each CSV column has a corresponding indicator."""
        for filename, shortname in BENCHMARK_FILES.items():
            filepath = benchmark_dir / filename
            if not filepath.exists():
                pytest.skip(f"Benchmark file not found: {filename}")

            # Read CSV to get column names
            df = pd.read_csv(filepath, index_col=0)
            columns = df.columns.tolist()

            # Check each column has an indicator
            for col in columns:
                indicator_name = f"{shortname}_{col}"
                row = db_connection.execute(
                    "SELECT indicator_name FROM indicators WHERE indicator_name = ?",
                    (indicator_name,)
                ).fetchone()

                assert row is not None, f"Missing indicator: {indicator_name}"

    def test_expected_indicator_count(self, loaded_benchmarks, benchmark_dir, db_connection):
        """Verify total number of benchmark indicators matches expectation."""
        # Count expected indicators from CSVs
        expected_count = 0
        for filename, shortname in BENCHMARK_FILES.items():
            filepath = benchmark_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0)
                expected_count += len(df.columns)

        # Count actual benchmark indicators
        row = db_connection.execute(
            "SELECT COUNT(*) FROM indicators WHERE system = 'benchmark'"
        ).fetchone()
        actual_count = row[0]

        assert actual_count >= expected_count, (
            f"Expected at least {expected_count} benchmark indicators, found {actual_count}"
        )


# =============================================================================
# TEST: TIMESERIES ROW COUNTS MATCH CSV ROW COUNTS
# =============================================================================

class TestTimeseriesRowCounts:
    """Test that timeseries data row counts match CSV row counts."""

    def test_row_counts_match_csv(self, loaded_benchmarks, benchmark_dir, db_connection):
        """Verify each indicator has the same number of rows as its source CSV."""
        for filename, shortname in BENCHMARK_FILES.items():
            filepath = benchmark_dir / filename
            if not filepath.exists():
                continue

            # Read CSV to get row count
            df = pd.read_csv(filepath, index_col=0)
            expected_rows = len(df)

            # Check each column's row count
            for col in df.columns:
                indicator_name = f"{shortname}_{col}"

                # Get indicator_id first
                ind_row = db_connection.execute(
                    "SELECT id FROM indicators WHERE indicator_name = ?",
                    (indicator_name,)
                ).fetchone()

                if ind_row is None:
                    pytest.fail(f"Indicator not found: {indicator_name}")

                indicator_id = ind_row[0]

                row = db_connection.execute(
                    "SELECT COUNT(*) FROM timeseries WHERE indicator_id = ?",
                    (indicator_id,)
                ).fetchone()
                actual_rows = row[0]

                assert actual_rows == expected_rows, (
                    f"Indicator {indicator_name}: expected {expected_rows} rows, got {actual_rows}"
                )

    def test_total_benchmark_rows(self, loaded_benchmarks, benchmark_dir, db_connection):
        """Verify total benchmark rows is reasonable."""
        row = db_connection.execute(
            """
            SELECT COUNT(*) FROM timeseries
            WHERE indicator_id IN (
                SELECT id FROM indicators WHERE system = 'benchmark'
            )
            """
        ).fetchone()
        total_rows = row[0]

        # Should have thousands of rows (6 files * ~6 columns * 1000 rows each)
        assert total_rows > 1000, f"Expected > 1000 benchmark rows, got {total_rows}"


# =============================================================================
# TEST: SYSTEM == "BENCHMARK"
# =============================================================================

class TestBenchmarkSystem:
    """Test that all benchmark indicators have system='benchmark'."""

    def test_all_have_benchmark_system(self, loaded_benchmarks, benchmark_dir, db_connection):
        """Verify all benchmark indicators have system='benchmark'."""
        for filename, shortname in BENCHMARK_FILES.items():
            filepath = benchmark_dir / filename
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath, index_col=0)

            for col in df.columns:
                indicator_name = f"{shortname}_{col}"
                row = db_connection.execute(
                    "SELECT system FROM indicators WHERE indicator_name = ?",
                    (indicator_name,)
                ).fetchone()

                assert row is not None, f"Indicator not found: {indicator_name}"
                assert row[0] == "benchmark", (
                    f"Indicator {indicator_name}: expected system='benchmark', got '{row[0]}'"
                )

    def test_no_benchmark_indicators_with_wrong_system(self, loaded_benchmarks, db_connection):
        """Verify no benchmark-pattern indicators exist outside system='benchmark'."""
        # Check for indicators with BENCH_ pattern in wrong system
        rows = db_connection.execute(
            """
            SELECT indicator_name, system FROM indicators
            WHERE (indicator_name LIKE '%clear_leader%'
                   OR indicator_name LIKE '%two_regimes%'
                   OR indicator_name LIKE '%clusters%'
                   OR indicator_name LIKE '%periodic%'
                   OR indicator_name LIKE '%anomalies%'
                   OR indicator_name LIKE '%pure_noise%')
            AND system != 'benchmark'
            """
        ).fetchall()

        assert len(rows) == 0, (
            f"Found benchmark-pattern indicators in wrong system: {[dict(r) for r in rows]}"
        )


# =============================================================================
# TEST: FRED_CODE STARTS WITH "BENCH_"
# =============================================================================

class TestFredCodePrefix:
    """Test that benchmark indicators have BENCH_ prefix in fred_code."""

    def test_fred_code_has_bench_prefix(self, loaded_benchmarks, benchmark_dir, db_connection):
        """Verify fred_code starts with BENCH_."""
        for filename, shortname in BENCHMARK_FILES.items():
            filepath = benchmark_dir / filename
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath, index_col=0)

            for col in df.columns:
                indicator_name = f"{shortname}_{col}"
                row = db_connection.execute(
                    "SELECT fred_code FROM indicators WHERE indicator_name = ?",
                    (indicator_name,)
                ).fetchone()

                assert row is not None, f"Indicator not found: {indicator_name}"
                fred_code = row[0]
                assert fred_code.startswith("BENCH_"), (
                    f"Indicator {indicator_name}: fred_code should start with 'BENCH_', got '{fred_code}'"
                )

    def test_fred_code_format(self, loaded_benchmarks, db_connection):
        """Verify fred_code follows expected format: BENCH_{shortname}_{column}."""
        rows = db_connection.execute(
            "SELECT indicator_name, fred_code FROM indicators WHERE system = 'benchmark'"
        ).fetchall()

        for row in rows:
            indicator_name = row[0]
            fred_code = row[1]

            # fred_code should be BENCH_{indicator_name}
            expected_fred_code = f"BENCH_{indicator_name}"
            assert fred_code == expected_fred_code, (
                f"Indicator {indicator_name}: expected fred_code '{expected_fred_code}', got '{fred_code}'"
            )


# =============================================================================
# TEST: NO NULL INDICATOR_NAME
# =============================================================================

class TestNoNullIndicatorName:
    """Test that no benchmark data has NULL indicator_name."""

    def test_no_null_indicator_names_in_indicators(self, loaded_benchmarks, db_connection):
        """Verify no NULL names in indicators table."""
        row = db_connection.execute(
            "SELECT COUNT(*) FROM indicators WHERE indicator_name IS NULL AND system = 'benchmark'"
        ).fetchone()

        assert row[0] == 0, "Found benchmark indicators with NULL indicator_name"

    def test_no_null_indicator_ids_in_timeseries(self, loaded_benchmarks, db_connection):
        """Verify no NULL indicator_id in timeseries for benchmarks."""
        row = db_connection.execute(
            "SELECT COUNT(*) FROM timeseries WHERE indicator_id IS NULL"
        ).fetchone()

        assert row[0] == 0, "Found timeseries rows with NULL indicator_id"

    def test_all_benchmark_timeseries_have_valid_indicator(self, loaded_benchmarks, db_connection):
        """Verify all benchmark timeseries reference valid indicators."""
        rows = db_connection.execute(
            """
            SELECT DISTINCT t.indicator_id
            FROM timeseries t
            LEFT JOIN indicators i ON t.indicator_id = i.id
            WHERE i.system = 'benchmark'
            AND i.id IS NULL
            """
        ).fetchall()

        orphaned = [r[0] for r in rows if r[0]]
        assert len(orphaned) == 0, f"Found orphaned benchmark timeseries: {orphaned}"


# =============================================================================
# TEST: DATA QUALITY
# =============================================================================

class TestDataQuality:
    """Additional tests for benchmark data quality."""

    def test_dates_are_valid_format(self, loaded_benchmarks, db_connection):
        """Verify all dates are in YYYY-MM-DD format."""
        rows = db_connection.execute(
            """
            SELECT DISTINCT date FROM timeseries
            WHERE indicator_id IN (
                SELECT id FROM indicators WHERE system = 'benchmark'
            )
            AND date NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
            """
        ).fetchall()

        invalid_dates = [r[0] for r in rows]
        assert len(invalid_dates) == 0, f"Found invalid date formats: {invalid_dates[:10]}"

    def test_values_are_numeric(self, loaded_benchmarks, db_connection):
        """Verify all values are numeric (not NULL for benchmarks)."""
        row = db_connection.execute(
            """
            SELECT COUNT(*) FROM timeseries
            WHERE indicator_id IN (
                SELECT id FROM indicators WHERE system = 'benchmark'
            )
            AND value IS NULL
            """
        ).fetchone()

        assert row[0] == 0, f"Found {row[0]} NULL values in benchmark data"

    def test_value_2_and_adjusted_are_null(self, loaded_benchmarks, db_connection):
        """Verify value_2 and adjusted_value are NULL for benchmarks."""
        row = db_connection.execute(
            """
            SELECT COUNT(*) FROM timeseries
            WHERE indicator_id IN (
                SELECT id FROM indicators WHERE system = 'benchmark'
            )
            AND (value_2 IS NOT NULL OR adjusted_value IS NOT NULL)
            """
        ).fetchone()

        assert row[0] == 0, f"Found {row[0]} rows with non-NULL value_2 or adjusted_value"


# =============================================================================
# TEST: LOADER FUNCTIONALITY
# =============================================================================

class TestLoaderFunctionality:
    """Test the loader module itself."""

    def test_load_all_benchmarks_returns_stats(self, benchmark_dir):
        """Verify load_all_benchmarks returns proper stats."""
        if not benchmark_dir.exists():
            pytest.skip("Benchmark directory not found")

        stats = load_all_benchmarks(verbose=False)

        assert "files_loaded" in stats
        assert "indicators_created" in stats
        assert "rows_inserted" in stats
        assert "errors" in stats

        assert stats["files_loaded"] > 0, "No files were loaded"
        assert stats["indicators_created"] > 0, "No indicators were created"
        assert stats["rows_inserted"] > 0, "No rows were inserted"

    def test_benchmark_files_mapping(self):
        """Verify BENCHMARK_FILES constant is complete."""
        assert len(BENCHMARK_FILES) == 6, "Expected 6 benchmark files in mapping"

        expected_shortnames = {
            "clear_leader", "two_regimes", "clusters",
            "periodic", "anomalies", "pure_noise"
        }
        actual_shortnames = set(BENCHMARK_FILES.values())

        assert actual_shortnames == expected_shortnames, (
            f"Unexpected shortnames: {actual_shortnames ^ expected_shortnames}"
        )


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestBenchmarkIntegration:
    """Integration tests for the complete benchmark system."""

    def test_can_query_benchmark_data(self, loaded_benchmarks, db_connection):
        """Verify benchmark data can be queried successfully."""
        # Simple query to fetch some benchmark data
        rows = db_connection.execute(
            """
            SELECT i.indicator_name, i.system, COUNT(t.date) as row_count
            FROM indicators i
            LEFT JOIN timeseries t ON i.id = t.indicator_id
            WHERE i.system = 'benchmark'
            GROUP BY i.indicator_name
            LIMIT 5
            """
        ).fetchall()

        assert len(rows) > 0, "No benchmark data found in query"

        for row in rows:
            assert row[0] is not None, "NULL indicator_name in query results"
            assert row[1] == "benchmark", f"Wrong system: {row[1]}"
            assert row[2] > 0, f"No data for indicator: {row[0]}"

    def test_benchmark_summary_stats(self, loaded_benchmarks, db_connection):
        """Generate and verify benchmark summary statistics."""
        # Get system counts
        row = db_connection.execute(
            "SELECT system, COUNT(*) FROM indicators GROUP BY system ORDER BY system"
        ).fetchall()

        systems = {r[0]: r[1] for r in row}

        assert "benchmark" in systems, "No benchmark system found"
        assert systems["benchmark"] >= 30, (
            f"Expected at least 30 benchmark indicators, found {systems['benchmark']}"
        )
