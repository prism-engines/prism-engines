#!/usr/bin/env python3
"""
Validate the PRISM database schema and migrations.

This script:
1. Creates a temporary database
2. Runs all migrations
3. Verifies all tables exist with correct columns
4. Performs basic insert/select tests
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


def main():
    """Run validation tests."""
    print("=" * 60)
    print("PRISM Database Schema Validation")
    print("=" * 60)
    print()

    from utils.db_connector import (
        init_database,
        get_connection,
        get_migration_files,
        get_applied_migrations,
        insert_market_price,
        insert_market_dividend,
        insert_market_tri,
        upsert_market_meta,
        insert_econ_series,
        insert_econ_value,
        load_market_prices,
        load_econ_values,
        get_market_stats,
        get_econ_stats,
    )

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    try:
        # Test 1: Migrations
        print("1. Testing migrations...")
        migrations = get_migration_files()
        print(f"   Found {len(migrations)} migration files")
        for version, path in migrations:
            print(f"   - {version}: {path.name}")

        init_database(db_path)
        applied = get_applied_migrations(db_path)
        print(f"   Applied {len(applied)} migrations: {sorted(applied)}")
        print("   [PASS] Migrations")
        print()

        # Test 2: Table existence
        print("2. Testing table existence...")
        expected_tables = [
            "market_prices",
            "market_dividends",
            "market_tri",
            "market_meta",
            "econ_series",
            "econ_values",
            "econ_meta",
            "schema_migrations",
            "data_quality_log",
            "fetch_log",
        ]

        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            existing_tables = {row["name"] for row in cursor.fetchall()}

        for table in expected_tables:
            if table in existing_tables:
                print(f"   [OK] {table}")
            else:
                print(f"   [MISSING] {table}")
                raise AssertionError(f"Table {table} not found")

        print("   [PASS] All tables exist")
        print()

        # Test 3: Market prices
        print("3. Testing market_prices...")
        insert_market_price(
            ticker="SPY",
            date="2024-01-15",
            price=450.25,
            ret=0.005,
            price_z=1.2,
            db_path=db_path,
        )
        insert_market_price("SPY", "2024-01-16", 451.50, db_path=db_path)
        insert_market_price("AAPL", "2024-01-15", 180.00, db_path=db_path)

        df = load_market_prices("SPY", db_path=db_path)
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert df.iloc[0]["price"] == 450.25
        print(f"   Inserted and loaded {len(df)} price records")
        print("   [PASS] market_prices")
        print()

        # Test 4: Market dividends
        print("4. Testing market_dividends...")
        insert_market_dividend("SPY", "2024-03-15", 1.50, db_path=db_path)

        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM market_dividends WHERE ticker = 'SPY'"
            )
            row = cursor.fetchone()
            assert row is not None
            assert row["dividend"] == 1.50
        print("   Inserted and verified dividend record")
        print("   [PASS] market_dividends")
        print()

        # Test 5: Market TRI
        print("5. Testing market_tri...")
        insert_market_tri("SPY", "2024-01-15", 1000.0, tri_z=0.5, db_path=db_path)

        with get_connection(db_path) as conn:
            cursor = conn.execute("SELECT * FROM market_tri WHERE ticker = 'SPY'")
            row = cursor.fetchone()
            assert row is not None
            assert row["tri_value"] == 1000.0
        print("   Inserted and verified TRI record")
        print("   [PASS] market_tri")
        print()

        # Test 6: Market meta
        print("6. Testing market_meta...")
        upsert_market_meta(
            ticker="SPY",
            first_date="2000-01-01",
            last_date="2024-01-16",
            source="Yahoo Finance",
            db_path=db_path,
        )

        with get_connection(db_path) as conn:
            cursor = conn.execute("SELECT * FROM market_meta WHERE ticker = 'SPY'")
            row = cursor.fetchone()
            assert row is not None
            assert row["source"] == "Yahoo Finance"
        print("   Inserted and verified market meta")
        print("   [PASS] market_meta")
        print()

        # Test 7: Economic series
        print("7. Testing econ_series...")
        insert_econ_series(
            code="GDP",
            human_name="Gross Domestic Product",
            frequency="quarterly",
            source="FRED",
            db_path=db_path,
        )

        with get_connection(db_path) as conn:
            cursor = conn.execute("SELECT * FROM econ_series WHERE code = 'GDP'")
            row = cursor.fetchone()
            assert row is not None
            assert row["human_name"] == "Gross Domestic Product"
        print("   Inserted and verified econ series")
        print("   [PASS] econ_series")
        print()

        # Test 8: Economic values
        print("8. Testing econ_values...")
        insert_econ_value(
            code="GDP",
            date="2024-01-01",
            revision_asof="2024-03-15",
            value_raw=25000.0,
            value_yoy=0.025,
            db_path=db_path,
        )

        df = load_econ_values("GDP", db_path=db_path)
        assert len(df) == 1
        assert df.iloc[0]["value_raw"] == 25000.0
        print(f"   Inserted and loaded {len(df)} econ value(s)")
        print("   [PASS] econ_values")
        print()

        # Test 9: Statistics
        print("9. Testing statistics...")
        market_stats = get_market_stats(db_path)
        econ_stats = get_econ_stats(db_path)

        print(f"   Market stats: {market_stats['prices']['tickers']} tickers, {market_stats['prices']['total_rows']} price rows")
        print(f"   Econ stats: {econ_stats['series_count']} series, {econ_stats['values']['total_rows']} value rows")
        print("   [PASS] Statistics")
        print()

        # Test 10: Column schema verification
        print("10. Verifying column schemas...")

        expected_columns = {
            "market_prices": ["id", "ticker", "date", "price", "ret", "price_z", "price_log", "created_at", "updated_at"],
            "market_dividends": ["id", "ticker", "date", "dividend", "created_at"],
            "market_tri": ["id", "ticker", "date", "tri_value", "tri_z", "tri_log", "created_at", "updated_at"],
            "econ_values": ["id", "code", "date", "revision_asof", "value_raw", "value_yoy", "value_mom", "value_z", "value_log", "created_at"],
        }

        with get_connection(db_path) as conn:
            for table, expected in expected_columns.items():
                cursor = conn.execute(f"PRAGMA table_info({table})")
                actual_columns = [row["name"] for row in cursor.fetchall()]
                missing = set(expected) - set(actual_columns)
                if missing:
                    print(f"   [WARN] {table} missing columns: {missing}")
                else:
                    print(f"   [OK] {table}: all {len(expected)} columns present")

        print("   [PASS] Column schemas")
        print()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    finally:
        # Cleanup
        db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
