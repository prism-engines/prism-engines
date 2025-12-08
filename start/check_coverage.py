#!/usr/bin/env python3
"""
Quick diagnostic: Check data coverage over time

Uses the unified indicator_values table (domain-agnostic).
"""
import pandas as pd
import sqlite3
from pathlib import Path

# === PRISM PATH CONFIG ===
import os
import sys
# Double-click support: ensure correct directory and path
if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
from output_config import OUTPUT_DIR, DATA_DIR
from data.sql.db_path import get_db_path
# === END PATH CONFIG ===


DB_PATH = get_db_path()

conn = sqlite3.connect(DB_PATH)

# Check indicator_values coverage (unified table)
print("=" * 60)
print("INDICATOR DATA COVERAGE (unified indicator_values table)")
print("=" * 60)

try:
    indicators = pd.read_sql(
        "SELECT indicator_name, date FROM indicator_values",
        conn
    )
    indicators['date'] = pd.to_datetime(indicators['date'])

    print(f"\nIndicators: {indicators['indicator_name'].nunique()}")
    print(f"Date range: {indicators['date'].min().date()} to {indicators['date'].max().date()}")

    print("\nFirst/last year each indicator appears:")
    for indicator in sorted(indicators['indicator_name'].unique()):
        subset = indicators[indicators['indicator_name'] == indicator]
        first_date = subset['date'].min()
        last_date = subset['date'].max()
        count = len(subset)
        print(f"  {indicator:25}: {first_date.year}-{last_date.year} ({count:,} obs)")

    # Check how many indicators available at different time points
    print("\n" + "=" * 60)
    print("INDICATORS AVAILABLE AT KEY DATES")
    print("=" * 60)

    key_dates = ['1987-10-19', '1998-08-17', '2000-03-10', '2008-09-15', '2020-03-16', '2025-04-02']
    for date_str in key_dates:
        date = pd.Timestamp(date_str)
        n_indicators = len(indicators[indicators['date'] <= date]['indicator_name'].unique())
        print(f"  {date_str}: {n_indicators} indicators")

except Exception as e:
    print(f"Error reading indicator_values: {e}")
    print("\nFalling back to legacy tables (deprecated)...")

    # Legacy fallback
    try:
        market = pd.read_sql("SELECT date, ticker FROM market_prices", conn)
        market['date'] = pd.to_datetime(market['date'])
        print(f"\nLegacy market_prices: {market['ticker'].nunique()} tickers")

        econ = pd.read_sql("SELECT date, series_id FROM econ_values", conn)
        econ['date'] = pd.to_datetime(econ['date'])
        print(f"Legacy econ_values: {econ['series_id'].nunique()} series")

        print("\nWARNING: Data is in legacy tables. Consider migrating to indicator_values.")
    except Exception as e2:
        print(f"Legacy tables also failed: {e2}")

conn.close()
