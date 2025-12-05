"""
Tests for the synthetic pipeline.

Tests:
- In-memory SQLite with sample data
- Run builder and assert expected rows exist
- Individual computation functions
"""

import pytest
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.sql.synthetic_pipeline import (
    build_synthetic_timeseries,
    load_dependencies,
    write_synthetic_series,
    compute_t10y2y,
    compute_t10y3m,
    compute_credit_spread,
    compute_yield_curve_slope,
    SYNTHETIC_COMPUTATIONS,
)


@pytest.fixture
def sample_registry():
    """Create a sample registry for testing."""
    return {
        "version": "test",
        "market": [
            {"name": "spy", "type": "market_price", "frequency": "daily"},
        ],
        "economic": [
            {"name": "dgs10", "type": "econ_series", "frequency": "daily"},
            {"name": "dgs2", "type": "econ_series", "frequency": "daily"},
            {"name": "dgs3mo", "type": "econ_series", "frequency": "daily"},
            {"name": "baaffm", "type": "econ_series", "frequency": "monthly"},
            {"name": "cpi", "type": "econ_series", "frequency": "monthly"},
            {"name": "ppi", "type": "econ_series", "frequency": "monthly"},
            {"name": "unrate", "type": "econ_series", "frequency": "monthly"},
            {"name": "payems", "type": "econ_series", "frequency": "monthly"},
            {"name": "m2", "type": "econ_series", "frequency": "monthly"},
            {"name": "gdp", "type": "econ_series", "frequency": "quarterly"},
        ],
        "synthetic": [
            {"name": "t10y2y", "depends_on": ["dgs10", "dgs2"]},
            {"name": "t10y3m", "depends_on": ["dgs10", "dgs3mo"]},
            {"name": "credit_spread", "depends_on": ["baaffm", "dgs10"]},
            {"name": "yield_curve_slope", "depends_on": ["dgs10", "dgs2"]},
        ],
    }


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Create schema
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            system TEXT DEFAULT 'finance',
            frequency TEXT DEFAULT 'daily',
            source TEXT,
            units TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS indicator_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            value REAL,
            value_2 REAL,
            adjusted_value REAL,
            FOREIGN KEY (indicator_id) REFERENCES indicators(id) ON DELETE CASCADE,
            UNIQUE(indicator_id, date)
        );

        CREATE INDEX IF NOT EXISTS idx_indicator_values_date
            ON indicator_values(indicator_id, date);
    """)
    conn.commit()

    yield conn

    conn.close()


@pytest.fixture
def populated_db(in_memory_db, sample_registry):
    """Populate the in-memory database with sample data."""
    conn = in_memory_db
    cursor = conn.cursor()

    # Create indicators
    indicators = {
        "dgs10": "daily",
        "dgs2": "daily",
        "dgs3mo": "daily",
        "baaffm": "monthly",
        "cpi": "monthly",
        "ppi": "monthly",
        "unrate": "monthly",
        "payems": "monthly",
        "m2": "monthly",
        "gdp": "quarterly",
    }

    indicator_ids = {}
    for name, freq in indicators.items():
        cursor.execute(
            "INSERT INTO indicators (name, system, frequency) VALUES (?, ?, ?)",
            (name, "finance", freq),
        )
        indicator_ids[name] = cursor.lastrowid

    # Generate sample time series data
    base_date = datetime(2023, 1, 1)
    num_days = 365

    # Daily data for treasury rates
    daily_data = {
        "dgs10": np.random.uniform(3.5, 4.5, num_days),  # 10Y around 4%
        "dgs2": np.random.uniform(4.0, 5.0, num_days),   # 2Y higher (inverted curve)
        "dgs3mo": np.random.uniform(4.5, 5.5, num_days), # 3M highest
        "baaffm": np.random.uniform(5.0, 6.0, num_days), # BAA spread
    }

    for name, values in daily_data.items():
        indicator_id = indicator_ids[name]
        for i, value in enumerate(values):
            date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            cursor.execute(
                "INSERT INTO indicator_values (indicator_id, date, value) VALUES (?, ?, ?)",
                (indicator_id, date, float(value)),
            )

    conn.commit()
    return conn, sample_registry


class TestComputeFunctions:
    """Test individual computation functions."""

    def test_compute_t10y2y(self):
        """Test 10Y-2Y spread computation."""
        df = pd.DataFrame({
            "dgs10": [4.0, 4.1, 4.2],
            "dgs2": [4.5, 4.6, 4.7],
        })
        result = compute_t10y2y(df)
        expected = pd.Series([-0.5, -0.5, -0.5])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_compute_t10y3m(self):
        """Test 10Y-3M spread computation."""
        df = pd.DataFrame({
            "dgs10": [4.0, 4.1, 4.2],
            "dgs3mo": [5.0, 5.1, 5.2],
        })
        result = compute_t10y3m(df)
        expected = pd.Series([-1.0, -1.0, -1.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_compute_credit_spread(self):
        """Test credit spread computation."""
        df = pd.DataFrame({
            "baaffm": [6.0, 6.1, 6.2],
            "dgs10": [4.0, 4.1, 4.2],
        })
        result = compute_credit_spread(df)
        expected = pd.Series([2.0, 2.0, 2.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_compute_yield_curve_slope(self):
        """Test yield curve slope computation."""
        df = pd.DataFrame({
            "dgs10": [4.0, 4.5, 5.0],
            "dgs2": [4.5, 4.5, 4.5],
        })
        result = compute_yield_curve_slope(df)
        expected = pd.Series([-0.5, 0.0, 0.5])
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestLoadDependencies:
    """Test loading dependencies from database."""

    def test_load_single_dependency(self, populated_db):
        """Test loading a single dependency."""
        conn, registry = populated_db
        deps = load_dependencies(conn, ["dgs10"])

        assert not deps.empty
        assert "dgs10" in deps.columns
        assert len(deps) > 0

    def test_load_multiple_dependencies(self, populated_db):
        """Test loading multiple dependencies."""
        conn, registry = populated_db
        deps = load_dependencies(conn, ["dgs10", "dgs2"])

        assert not deps.empty
        assert "dgs10" in deps.columns
        assert "dgs2" in deps.columns
        assert len(deps) > 0

    def test_load_with_date_filter(self, populated_db):
        """Test loading with date filter."""
        conn, registry = populated_db
        deps = load_dependencies(
            conn,
            ["dgs10"],
            start_date="2023-06-01",
            end_date="2023-06-30",
        )

        assert not deps.empty
        assert all(deps.index >= pd.Timestamp("2023-06-01"))
        assert all(deps.index <= pd.Timestamp("2023-06-30"))

    def test_load_nonexistent_dependency(self, populated_db):
        """Test loading a nonexistent dependency."""
        conn, registry = populated_db
        deps = load_dependencies(conn, ["nonexistent"])

        assert deps.empty


class TestWriteSyntheticSeries:
    """Test writing synthetic series to database."""

    def test_write_creates_indicator(self, in_memory_db):
        """Test that write creates indicator if not exists."""
        conn = in_memory_db

        # Create test data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        data = pd.DataFrame({"value": range(10)}, index=dates)

        rows = write_synthetic_series(conn, "test_synthetic", data)

        assert rows == 10

        # Check indicator was created
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM indicators WHERE name = 'test_synthetic'")
        row = cursor.fetchone()
        assert row is not None
        assert row["source"] == "synthetic"

    def test_write_updates_existing(self, in_memory_db):
        """Test that write updates existing values."""
        conn = in_memory_db

        # Create test data
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data1 = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)
        data2 = pd.DataFrame({"value": [10, 20, 30, 40, 50]}, index=dates)

        # Write first version
        write_synthetic_series(conn, "test_synthetic", data1)

        # Write second version (should update)
        write_synthetic_series(conn, "test_synthetic", data2)

        # Check values were updated
        cursor = conn.cursor()
        cursor.execute("""
            SELECT iv.value FROM indicator_values iv
            JOIN indicators i ON iv.indicator_id = i.id
            WHERE i.name = 'test_synthetic' AND iv.date = '2023-01-01'
        """)
        row = cursor.fetchone()
        assert row["value"] == 10.0


class TestBuildSyntheticTimeseries:
    """Test the main build function."""

    def test_build_creates_synthetics(self, populated_db):
        """Test that build creates synthetic time series."""
        conn, registry = populated_db

        results = build_synthetic_timeseries(registry, conn)

        # Check that results were returned
        assert isinstance(results, dict)
        assert "t10y2y" in results
        assert "t10y3m" in results
        assert "credit_spread" in results
        assert "yield_curve_slope" in results

    def test_build_writes_to_database(self, populated_db):
        """Test that build writes data to database."""
        conn, registry = populated_db

        results = build_synthetic_timeseries(registry, conn)

        # Check data was written
        cursor = conn.cursor()
        for name in ["t10y2y", "t10y3m", "credit_spread", "yield_curve_slope"]:
            if results.get(name, 0) > 0:
                cursor.execute("""
                    SELECT COUNT(*) as count FROM indicator_values iv
                    JOIN indicators i ON iv.indicator_id = i.id
                    WHERE i.name = ?
                """, (name,))
                row = cursor.fetchone()
                assert row["count"] > 0, f"No data written for {name}"

    def test_build_returns_row_counts(self, populated_db):
        """Test that build returns accurate row counts."""
        conn, registry = populated_db

        results = build_synthetic_timeseries(registry, conn)

        # At least some synthetics should have data
        total_rows = sum(results.values())
        assert total_rows > 0

    def test_build_handles_missing_data(self, in_memory_db, sample_registry):
        """Test that build handles missing dependencies gracefully."""
        conn = in_memory_db

        # Don't populate any data - dependencies will be missing
        results = build_synthetic_timeseries(sample_registry, conn)

        # Should return zeros for all metrics (no data)
        for name, rows in results.items():
            assert rows == 0


class TestSyntheticComputationRegistry:
    """Test the computation function registry."""

    def test_all_synthetics_have_compute_functions(self, sample_registry):
        """Test that all synthetic metrics have computation functions."""
        synthetic_metrics = sample_registry.get("synthetic", [])

        for metric in synthetic_metrics:
            name = metric.get("name")
            assert name in SYNTHETIC_COMPUTATIONS, (
                f"No computation function for synthetic '{name}'"
            )

    def test_compute_functions_callable(self):
        """Test that all computation functions are callable."""
        for name, func in SYNTHETIC_COMPUTATIONS.items():
            assert callable(func), f"Computation function for '{name}' is not callable"

    def test_compute_functions_return_series(self):
        """Test that computation functions return pandas Series."""
        # Create sample input
        df = pd.DataFrame({
            "dgs10": [4.0, 4.1, 4.2],
            "dgs2": [4.5, 4.6, 4.7],
            "dgs3mo": [5.0, 5.1, 5.2],
            "baaffm": [6.0, 6.1, 6.2],
            "cpi": [300, 301, 302],
            "ppi": [200, 201, 202],
            "unrate": [3.5, 3.6, 3.7],
            "payems": [150000, 150100, 150200],
            "m2": [21000, 21100, 21200],
            "gdp": [25000, 25000, 25000],
        })

        for name, func in SYNTHETIC_COMPUTATIONS.items():
            try:
                result = func(df)
                assert isinstance(result, pd.Series), (
                    f"Computation '{name}' did not return Series"
                )
            except KeyError:
                # Some functions may need specific columns
                pass
