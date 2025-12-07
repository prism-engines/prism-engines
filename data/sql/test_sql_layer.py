#!/usr/bin/env python3
"""
test_sql_layer.py - Sanity checks for the unified SQL layer (Schema v2).

This script validates that:
1. All modules import without errors
2. connect() returns a connection
3. init_database() creates DB tables (including indicator_values)
4. add_indicator(), get_indicator(), list_indicators() work
5. write_dataframe() writes to indicator_values (unified table)
6. load_indicator() returns data from indicator_values

Schema v2 uses indicator_values as the single source of truth.
Legacy tables (market_prices, econ_values) are deprecated.

Usage:
    python -m data.sql.test_sql_layer
    # OR
    python data/sql/test_sql_layer.py
"""

import os
import sys
import tempfile
import shutil

# Ensure we can import from project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_imports():
    """Test that all SQL modules import without errors."""
    print("Testing imports...")

    try:
        import data.sql
        print("  [OK] import data.sql")
    except ImportError as e:
        print(f"  [FAIL] import data.sql: {e}")
        return False

    try:
        import data.sql.db
        print("  [OK] import data.sql.db")
    except ImportError as e:
        print(f"  [FAIL] import data.sql.db: {e}")
        return False

    try:
        import data.sql.prism_db
        print("  [OK] import data.sql.prism_db")
    except ImportError as e:
        print(f"  [FAIL] import data.sql.prism_db: {e}")
        return False

    try:
        import data.sql.db_connector
        print("  [OK] import data.sql.db_connector")
    except ImportError as e:
        print(f"  [FAIL] import data.sql.db_connector: {e}")
        return False

    return True


def test_connection():
    """Test that connect() returns a working connection."""
    print("\nTesting connection...")

    from data.sql import connect, get_connection

    try:
        conn = connect()
        assert conn is not None, "connect() returned None"
        conn.close()
        print("  [OK] connect() returns a connection")
    except Exception as e:
        print(f"  [FAIL] connect(): {e}")
        return False

    try:
        conn = get_connection()
        assert conn is not None, "get_connection() returned None"
        conn.close()
        print("  [OK] get_connection() returns a connection")
    except Exception as e:
        print(f"  [FAIL] get_connection(): {e}")
        return False

    return True


def test_init_database():
    """Test that init_database() creates required tables."""
    print("\nTesting init_database...")

    from data.sql import init_database, get_connection

    try:
        init_database()
        print("  [OK] init_database() executed")
    except Exception as e:
        print(f"  [FAIL] init_database(): {e}")
        return False

    # Verify tables exist
    try:
        conn = get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        conn.close()

        required_tables = ["indicators", "market_prices", "econ_values", "fetch_log"]
        for t in required_tables:
            if t in table_names:
                print(f"  [OK] Table '{t}' exists")
            else:
                print(f"  [FAIL] Table '{t}' missing")
                return False

    except Exception as e:
        print(f"  [FAIL] Verifying tables: {e}")
        return False

    return True


def test_indicator_operations():
    """Test add_indicator, get_indicator, list_indicators."""
    print("\nTesting indicator operations...")

    from data.sql import add_indicator, get_indicator, list_indicators

    # Test add_indicator
    try:
        add_indicator("TEST_INDICATOR", system="market", metadata={"source": "test"})
        print("  [OK] add_indicator() executed")
    except Exception as e:
        print(f"  [FAIL] add_indicator(): {e}")
        return False

    # Test get_indicator
    try:
        indicator = get_indicator("TEST_INDICATOR")
        assert indicator is not None, "get_indicator() returned None"
        assert indicator["name"] == "TEST_INDICATOR", "Indicator name mismatch"
        assert indicator["system"] == "market", "Indicator system mismatch"
        print(f"  [OK] get_indicator() returned: {indicator['name']}")
    except Exception as e:
        print(f"  [FAIL] get_indicator(): {e}")
        return False

    # Test list_indicators
    try:
        indicators = list_indicators()
        assert isinstance(indicators, list), "list_indicators() did not return list"
        assert "TEST_INDICATOR" in indicators, "TEST_INDICATOR not in list"
        print(f"  [OK] list_indicators() returned {len(indicators)} indicators")
    except Exception as e:
        print(f"  [FAIL] list_indicators(): {e}")
        return False

    # Test list_indicators with filter
    try:
        market_indicators = list_indicators(system="market")
        assert "TEST_INDICATOR" in market_indicators
        print(f"  [OK] list_indicators(system='market') works")
    except Exception as e:
        print(f"  [FAIL] list_indicators(system='market'): {e}")
        return False

    return True


def test_write_dataframe():
    """Test write_dataframe for indicator_values (Schema v2)."""
    print("\nTesting write_dataframe...")

    import pandas as pd
    from data.sql import write_dataframe, init_database, add_indicator

    # Ensure tables exist
    init_database()

    # Register indicators first
    add_indicator("SPY", system="market", source="test")
    add_indicator("GDP", system="economic", source="test")

    # Test writing to indicator_values (Schema v2 - unified table)
    try:
        market_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [100.0, 101.0, 102.0]
        })
        rows = write_dataframe(market_df, "indicator_values", indicator_name="SPY", provenance="test")
        print(f"  [OK] write_dataframe() to indicator_values (SPY): {rows} rows")
    except Exception as e:
        print(f"  [FAIL] write_dataframe() to indicator_values (SPY): {e}")
        return False

    # Test writing another indicator
    try:
        econ_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [1000.0, 1001.0, 1002.0]
        })
        rows = write_dataframe(econ_df, "indicator_values", indicator_name="GDP", provenance="test")
        print(f"  [OK] write_dataframe() to indicator_values (GDP): {rows} rows")
    except Exception as e:
        print(f"  [FAIL] write_dataframe() to indicator_values (GDP): {e}")
        return False

    return True


def test_load_indicator():
    """Test load_indicator returns unified results from indicator_values."""
    print("\nTesting load_indicator...")

    from data.sql import load_indicator

    # Test loading market data
    try:
        df = load_indicator("SPY")
        assert len(df) > 0, "load_indicator('SPY') returned empty DataFrame"
        assert "indicator_name" in df.columns, "Missing 'indicator_name' column"
        assert "date" in df.columns, "Missing 'date' column"
        assert "value" in df.columns, "Missing 'value' column"
        print(f"  [OK] load_indicator('SPY') returned {len(df)} rows")
    except Exception as e:
        print(f"  [FAIL] load_indicator('SPY'): {e}")
        return False

    # Test loading econ data
    try:
        df = load_indicator("GDP")
        assert len(df) > 0, "load_indicator('GDP') returned empty DataFrame"
        print(f"  [OK] load_indicator('GDP') returned {len(df)} rows")
    except Exception as e:
        print(f"  [FAIL] load_indicator('GDP'): {e}")
        return False

    return True


def test_load_multiple_indicators():
    """Test load_multiple_indicators."""
    print("\nTesting load_multiple_indicators...")

    from data.sql import load_multiple_indicators

    try:
        df = load_multiple_indicators(["SPY", "GDP"])
        assert len(df) > 0, "load_multiple_indicators() returned empty DataFrame"
        unique_indicators = df["indicator_name"].unique()
        assert len(unique_indicators) == 2, f"Expected 2 indicators, got {len(unique_indicators)}"
        print(f"  [OK] load_multiple_indicators(['SPY', 'GDP']) returned {len(df)} rows")
    except Exception as e:
        print(f"  [FAIL] load_multiple_indicators(): {e}")
        return False

    return True


def test_statistics():
    """Test database_stats, get_table_stats, get_date_range."""
    print("\nTesting statistics functions...")

    from data.sql import database_stats, get_table_stats, get_date_range

    # Test database_stats
    try:
        stats = database_stats()
        assert "tables" in stats, "database_stats missing 'tables' key"
        assert "total_rows" in stats, "database_stats missing 'total_rows' key"
        print(f"  [OK] database_stats() returned {len(stats['tables'])} tables, {stats['total_rows']} total rows")
    except Exception as e:
        print(f"  [FAIL] database_stats(): {e}")
        return False

    # Test get_table_stats on unified table
    try:
        stats = get_table_stats("indicator_values")
        assert "row_count" in stats, "get_table_stats missing 'row_count' key"
        print(f"  [OK] get_table_stats('indicator_values') returned {stats['row_count']} rows")
    except Exception as e:
        print(f"  [FAIL] get_table_stats(): {e}")
        return False

    # Test get_date_range on unified table
    try:
        date_range = get_date_range(indicator_name="SPY")
        assert "min_date" in date_range, "get_date_range missing 'min_date' key"
        assert "max_date" in date_range, "get_date_range missing 'max_date' key"
        print(f"  [OK] get_date_range(indicator_name='SPY'): {date_range['min_date']} to {date_range['max_date']}")
    except Exception as e:
        print(f"  [FAIL] get_date_range(): {e}")
        return False

    return True


def test_log_fetch():
    """Test log_fetch and get_fetch_history."""
    print("\nTesting fetch logging...")

    from data.sql import log_fetch, get_fetch_history

    try:
        log_id = log_fetch(
            source="test",
            entity="TEST_ENTITY",
            operation="fetch",
            status="success",
            rows_fetched=100,
            rows_inserted=50
        )
        assert log_id > 0, "log_fetch returned invalid ID"
        print(f"  [OK] log_fetch() returned ID {log_id}")
    except Exception as e:
        print(f"  [FAIL] log_fetch(): {e}")
        return False

    try:
        history = get_fetch_history(entity="TEST_ENTITY", limit=10)
        assert len(history) > 0, "get_fetch_history returned empty DataFrame"
        print(f"  [OK] get_fetch_history() returned {len(history)} entries")
    except Exception as e:
        print(f"  [FAIL] get_fetch_history(): {e}")
        return False

    return True


def test_legacy_prism_db():
    """Test that prism_db works as a legacy module."""
    print("\nTesting legacy prism_db module...")

    from data.sql import prism_db

    # Test get_connection
    try:
        conn = prism_db.get_connection()
        assert conn is not None
        conn.close()
        print("  [OK] prism_db.get_connection() works")
    except Exception as e:
        print(f"  [FAIL] prism_db.get_connection(): {e}")
        return False

    # Test query
    try:
        df = prism_db.query("SELECT 1 as test")
        assert len(df) == 1
        print("  [OK] prism_db.query() works")
    except Exception as e:
        print(f"  [FAIL] prism_db.query(): {e}")
        return False

    return True


def run_all_tests():
    """Run all sanity checks."""
    print("=" * 60)
    print("SQL Layer Sanity Tests")
    print("=" * 60)

    # Use a temporary database for testing
    temp_dir = tempfile.mkdtemp()
    test_db_path = os.path.join(temp_dir, "test_prism.db")

    # Set environment variable to use test database
    os.environ["PRISM_DB_PATH"] = test_db_path
    print(f"\nUsing test database: {test_db_path}\n")

    tests = [
        ("Imports", test_imports),
        ("Connection", test_connection),
        ("Init Database", test_init_database),
        ("Indicator Operations", test_indicator_operations),
        ("Write DataFrame", test_write_dataframe),
        ("Load Indicator", test_load_indicator),
        ("Load Multiple Indicators", test_load_multiple_indicators),
        ("Statistics", test_statistics),
        ("Log Fetch", test_log_fetch),
        ("Legacy prism_db", test_legacy_prism_db),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[EXCEPTION] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    # Clear environment variable
    if "PRISM_DB_PATH" in os.environ:
        del os.environ["PRISM_DB_PATH"]

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
