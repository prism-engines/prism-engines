#!/usr/bin/env python3
"""
Quick diagnostic: Check data coverage over time
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
# === END PATH CONFIG ===


DB_PATH = DATA_DIR / "prism.db"

conn = sqlite3.connect(DB_PATH)

# Check market data coverage
market = pd.read_sql("SELECT date, ticker FROM market_prices", conn)
market['date'] = pd.to_datetime(market['date'])

print("=" * 60)
print("MARKET DATA COVERAGE BY YEAR")
print("=" * 60)
market_by_year = market.groupby([market['date'].dt.year, 'ticker']).size().unstack(fill_value=0)
print(f"\nTickers: {market['ticker'].nunique()}")
print(f"Date range: {market['date'].min().date()} to {market['date'].max().date()}")
print("\nFirst year each ticker appears:")
for ticker in market['ticker'].unique():
    first_date = market[market['ticker'] == ticker]['date'].min()
    print(f"  {ticker}: {first_date.year}")

# Check economic data coverage  
econ = pd.read_sql("SELECT date, series_id FROM econ_values", conn)
econ['date'] = pd.to_datetime(econ['date'])

print("\n" + "=" * 60)
print("ECONOMIC DATA COVERAGE BY YEAR")
print("=" * 60)
print(f"\nSeries: {econ['series_id'].nunique()}")
print(f"Date range: {econ['date'].min().date()} to {econ['date'].max().date()}")
print("\nFirst year each series appears:")
for series in sorted(econ['series_id'].unique()):
    first_date = econ[econ['series_id'] == series]['date'].min()
    last_date = econ[econ['series_id'] == series]['date'].max()
    count = len(econ[econ['series_id'] == series])
    print(f"  {series:20}: {first_date.year}-{last_date.year} ({count:,} obs)")

conn.close()

# Check how many indicators available at different time points
print("\n" + "=" * 60)
print("INDICATORS AVAILABLE AT KEY DATES")
print("=" * 60)

key_dates = ['1987-10-19', '1998-08-17', '2000-03-10', '2008-09-15', '2020-03-16', '2025-04-02']
for date_str in key_dates:
    date = pd.Timestamp(date_str)
    n_market = len(market[market['date'] <= date]['ticker'].unique())
    n_econ = len(econ[econ['date'] <= date]['series_id'].unique())
    print(f"  {date_str}: {n_market} market + {n_econ} economic = {n_market + n_econ} total")
