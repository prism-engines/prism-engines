#!/usr/bin/env python3
"""
PRISM Data Inventory Report
============================

Shows what data you have, coverage, gaps, and freshness.

Usage:
    python start/data_report.py
    python start/data_report.py --csv  # Export to CSV
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

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


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_db_connection():
    """Get database connection."""
    try:
        import sqlite3
        from data.sql.db_path import get_db_path

        db_path = get_db_path()
        print(f"Using database: {db_path}")
        return sqlite3.connect(db_path)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def get_market_data_report(conn):
    """Get report on market/price data."""
    
    # Try to find the prices table - could be named differently
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\nFound tables: {tables}")
    
    # Look for price-related tables
    price_tables = [t for t in tables if any(x in t.lower() for x in ['price', 'market', 'daily', 'ohlc'])]
    econ_tables = [t for t in tables if any(x in t.lower() for x in ['econ', 'fred', 'series'])]
    
    reports = []
    
    for table in price_tables:
        try:
            # Get column names
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"\n{table} columns: {columns}")
            
            # Figure out the grouping column (indicator_id, ticker, symbol, name)
            group_col = None
            for col in ['indicator_id', 'ticker', 'symbol', 'name', 'id']:
                if col in columns:
                    group_col = col
                    break
            
            date_col = None
            for col in ['date', 'timestamp', 'time']:
                if col in columns:
                    date_col = col
                    break
            
            if group_col and date_col:
                query = f"""
                SELECT 
                    {group_col} as indicator,
                    COUNT(*) as row_count,
                    MIN({date_col}) as first_date,
                    MAX({date_col}) as last_date
                FROM {table}
                GROUP BY {group_col}
                ORDER BY {group_col}
                """
                df = pd.read_sql(query, conn)
                df['table'] = table
                df['type'] = 'market'
                reports.append(df)
        except Exception as e:
            print(f"Error reading {table}: {e}")
    
    for table in econ_tables:
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"\n{table} columns: {columns}")
            
            group_col = None
            for col in ['series_id', 'indicator_id', 'series', 'name', 'id']:
                if col in columns:
                    group_col = col
                    break
            
            date_col = None
            for col in ['date', 'timestamp', 'time']:
                if col in columns:
                    date_col = col
                    break
            
            if group_col and date_col:
                query = f"""
                SELECT 
                    {group_col} as indicator,
                    COUNT(*) as row_count,
                    MIN({date_col}) as first_date,
                    MAX({date_col}) as last_date
                FROM {table}
                GROUP BY {group_col}
                ORDER BY {group_col}
                """
                df = pd.read_sql(query, conn)
                df['table'] = table
                df['type'] = 'economic'
                reports.append(df)
        except Exception as e:
            print(f"Error reading {table}: {e}")
    
    if reports:
        return pd.concat(reports, ignore_index=True)
    return pd.DataFrame()


def calculate_coverage(df):
    """Add coverage statistics to the report."""
    if df.empty:
        return df
    
    today = datetime.now().date()
    
    def calc_stats(row):
        try:
            first = pd.to_datetime(row['first_date']).date()
            last = pd.to_datetime(row['last_date']).date()
            
            # Trading days (approximate)
            calendar_days = (last - first).days
            expected_trading_days = int(calendar_days * 252 / 365)  # ~252 trading days/year
            
            # Freshness
            days_stale = (today - last).days
            
            # Coverage percentage
            coverage = min(100, (row['row_count'] / max(1, expected_trading_days)) * 100)
            
            return pd.Series({
                'years': round(calendar_days / 365, 1),
                'expected_rows': expected_trading_days,
                'coverage_pct': round(coverage, 1),
                'days_stale': days_stale,
                'status': '✅' if days_stale <= 3 else '⚠️' if days_stale <= 7 else '❌'
            })
        except:
            return pd.Series({
                'years': 0,
                'expected_rows': 0,
                'coverage_pct': 0,
                'days_stale': 999,
                'status': '❓'
            })
    
    stats = df.apply(calc_stats, axis=1)
    return pd.concat([df, stats], axis=1)


def print_report(df):
    """Print a nice formatted report."""
    
    if df.empty:
        print("\nNo data found in database!")
        return
    
    print("\n" + "=" * 80)
    print("PRISM DATA INVENTORY REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Total indicators: {len(df)}")
    print(f"Total rows: {df['row_count'].sum():,}")
    
    if 'type' in df.columns:
        for dtype in df['type'].unique():
            subset = df[df['type'] == dtype]
            print(f"  {dtype}: {len(subset)} indicators, {subset['row_count'].sum():,} rows")
    
    # Freshness summary
    if 'status' in df.columns:
        fresh = len(df[df['status'] == '✅'])
        stale = len(df[df['status'] == '⚠️'])
        old = len(df[df['status'] == '❌'])
        print(f"\nFreshness: {fresh} current, {stale} slightly stale, {old} old")
    
    # Detail by type
    for dtype in df['type'].unique() if 'type' in df.columns else ['all']:
        subset = df[df['type'] == dtype] if 'type' in df.columns else df
        
        print("\n" + "-" * 40)
        print(f"{dtype.upper()} DATA")
        print("-" * 40)
        
        # Format for display
        display_cols = ['indicator', 'row_count', 'first_date', 'last_date', 'years', 'coverage_pct', 'status']
        display_cols = [c for c in display_cols if c in subset.columns]
        
        display_df = subset[display_cols].copy()
        
        # Rename for readability
        display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]
        
        print(display_df.to_string(index=False))
    
    # Warnings
    if 'days_stale' in df.columns:
        stale_data = df[df['days_stale'] > 7]
        if not stale_data.empty:
            print("\n" + "-" * 40)
            print("⚠️  STALE DATA WARNING")
            print("-" * 40)
            for _, row in stale_data.iterrows():
                print(f"  {row['indicator']}: {row['days_stale']} days old (last: {row['last_date']})")


def export_csv(df, filename="data_inventory.csv"):
    """Export report to CSV."""
    output_path = PROJECT_ROOT / filename
    df.to_csv(output_path, index=False)
    print(f"\nExported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PRISM Data Inventory Report")
    parser.add_argument("--csv", action="store_true", help="Export to CSV")
    parser.add_argument("--output", type=str, default="data_inventory.csv", help="CSV output filename")
    args = parser.parse_args()
    
    conn = get_db_connection()
    if not conn:
        return 1
    
    try:
        # Get raw report
        df = get_market_data_report(conn)
        
        if df.empty:
            print("\nNo data found. Have you run the fetcher yet?")
            print("  python start/fetcher.py --all")
            return 1
        
        # Add coverage stats
        df = calculate_coverage(df)
        
        # Print report
        print_report(df)
        
        # Export if requested
        if args.csv:
            export_csv(df, args.output)
        
        return 0
        
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
