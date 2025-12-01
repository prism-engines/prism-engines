#!/usr/bin/env python3
"""
CSV to SQL Migration Script
===========================

Migrates existing CSV data from data/raw to the SQL database.

Usage:
    python data/sql/migrate_csv_to_sql.py

This script will:
1. Read all CSV files from data/raw/
2. Import them into the SQL database as raw data
3. Store cleaned versions after basic cleaning
4. Create a master panel from all data
"""

import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def migrate_csv_to_sql(raw_dir: Path = None, verbose: bool = True):
    """
    Migrate CSV files to SQL database.
    
    Args:
        raw_dir: Directory containing CSV files (default: data/raw)
        verbose: Print progress
    """
    from data.sql import SQLDataManager
    
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / 'data' / 'raw'
    
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        print(f"Error: Directory not found: {raw_dir}")
        return
    
    # Initialize database
    db = SQLDataManager()
    db.init_schema()
    
    if verbose:
        print("="*60)
        print("CSV to SQL Migration")
        print("="*60)
        print(f"Source: {raw_dir}")
        print(f"Database: {db.db_path}")
        print()
    
    # FRED series (for source detection)
    fred_tickers = {
        'dgs10', 'dgs2', 'dgs3mo', 't10y2y', 't10y3m',
        'cpiaucsl', 'cpilfesl', 'ppiaco',
        'unrate', 'payems',
        'indpro', 'houst', 'permit',
        'm2sl', 'walcl',
        'anfci', 'nfci'
    }
    
    # Find all CSV files
    csv_files = list(raw_dir.glob('*.csv'))
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files")
        print()
    
    success_count = 0
    error_count = 0
    all_data = {}
    
    for csv_file in csv_files:
        ticker = csv_file.stem.lower()
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            if df.empty:
                if verbose:
                    print(f"  ⚠ {ticker}: Empty file, skipping")
                continue
            
            # Reset index to get date column
            df = df.reset_index()
            df.columns = ['date'] + [c.lower() for c in df.columns[1:]]
            
            # Determine source
            source = 'fred' if ticker in fred_tickers else 'yahoo'
            
            # Determine category
            if ticker in fred_tickers:
                category = 'macro'
            elif ticker in ['spy', 'qqq', 'iwm']:
                category = 'equity'
            elif ticker in ['gld', 'slv', 'uso', 'bcom']:
                category = 'commodity'
            elif ticker in ['bnd', 'tlt', 'shy', 'ief', 'tip', 'lqd', 'hyg']:
                category = 'bonds'
            elif ticker in ['dxy']:
                category = 'currency'
            elif ticker in ['vix']:
                category = 'volatility'
            else:
                category = 'other'
            
            # Store raw data
            raw_count = db.store_raw_data(
                df, 
                source=source, 
                ticker=ticker,
                category=category
            )
            
            # Clean data (simple forward fill)
            df_clean = df.copy()
            for col in df_clean.columns:
                if col != 'date':
                    df_clean[col] = df_clean[col].ffill().bfill()
            
            # Store cleaned data
            value_col = None
            for col in df_clean.columns:
                if col != 'date' and col != 'id':
                    value_col = col
                    break
            
            if value_col:
                nan_before = df[value_col].isna().sum() if value_col in df.columns else 0
                db.store_cleaned_data(
                    df_clean,
                    ticker=ticker,
                    cleaning_method='ffill',
                    source_rows=len(df),
                    nan_before=nan_before
                )
                
                # Store for combined panel
                df_clean = df_clean.set_index('date')
                if value_col in df_clean.columns:
                    all_data[ticker] = df_clean[value_col]
            
            if verbose:
                print(f"  ✓ {ticker}: {raw_count} rows imported from {source}")
            
            success_count += 1
            
        except Exception as e:
            if verbose:
                print(f"  ✗ {ticker}: {e}")
            error_count += 1
    
    # Create combined panel
    if all_data:
        if verbose:
            print()
            print("Creating master panel...")
        
        # Filter out any non-numeric series
        numeric_data = {}
        for ticker, series in all_data.items():
            if pd.api.types.is_numeric_dtype(series):
                numeric_data[ticker] = series
            else:
                # Try to convert
                try:
                    numeric_data[ticker] = pd.to_numeric(series, errors='coerce')
                except Exception:
                    if verbose:
                        print(f"  ⚠ Skipping non-numeric: {ticker}")
        
        if numeric_data:
            combined = pd.DataFrame(numeric_data)
            combined = combined.ffill().bfill()
            combined = combined.dropna(how='all')
            
            db.store_panel(combined, name='master_panel', description='Migrated from CSV files')
            
            if verbose:
                print(f"  ✓ Master panel: {combined.shape[1]} indicators, {combined.shape[0]} rows")
    
    # Summary
    if verbose:
        print()
        print("="*60)
        print("Migration Complete")
        print("="*60)
        print(f"  Successful: {success_count}")
        print(f"  Errors: {error_count}")
        print()
        
        stats = db.get_stats()
        print("Database Statistics:")
        for table, count in stats.items():
            print(f"  {table}: {count} rows")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate CSV files to SQL database')
    parser.add_argument('--raw-dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    migrate_csv_to_sql(raw_dir=raw_dir, verbose=not args.quiet)


if __name__ == '__main__':
    main()
