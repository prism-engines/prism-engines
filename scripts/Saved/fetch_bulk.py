#!/usr/bin/env python3
"""
PRISM Bulk Fetcher

Fetches 100+ indicators with rate limiting.
Designed to run overnight on cloud compute.

Usage:
    python scripts/fetch_bulk.py
    python scripts/fetch_bulk.py --delay 2.0
    python scripts/fetch_bulk.py --resume
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from prism.db.connection import get_db_path, get_connection


def load_expanded_registry(path='prism/registry/indicators_expanded.yaml'):
    """Load expanded indicator registry."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get('indicators', [])


def get_fetched_indicators(db_path=None):
    """Get list of already fetched indicators."""
    try:
        conn = get_connection(Path(db_path) if db_path else None)
        result = conn.execute('''
            SELECT DISTINCT indicator_id 
            FROM clean.indicators
        ''').fetchdf()
        return set(result['indicator_id'].tolist())
    except:
        return set()


def fetch_single_indicator(indicator, delay=1.0):
    """Fetch a single indicator with error handling."""
    from prism.fetch.sources.fred import FREDFetcher
    from prism.fetch.sources.tiingo import TiingoFetcher
    from prism.cleaning.pipeline import CleaningPipeline
    from prism.db.connection import connect
    
    ind_id = indicator['id']
    source = indicator['source']
    
    try:
        # Get fetcher
        if source == 'fred':
            fetcher = FREDFetcher()
        elif source == 'tiingo':
            fetcher = TiingoFetcher()
        else:
            return False, f"Unknown source: {source}"
        
        # Fetch
        df = fetcher.fetch(ind_id)
        
        if df is None or len(df) == 0:
            return False, "No data returned"
        
        # Clean
        pipeline = CleaningPipeline()
        df_clean = pipeline.clean(df, ind_id)
        
        if df_clean is None or len(df_clean) == 0:
            return False, "Cleaning failed"
        
        # Store
        conn = connect()
        
        # Insert raw
        for _, row in df.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO raw.indicators 
                (date, indicator_id, source, value, fetched_at)
                VALUES (?, ?, ?, ?, ?)
            ''', [row['date'], ind_id, source, row['value'], datetime.now()])
        
        # Insert clean
        for _, row in df_clean.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO clean.indicators 
                (date, indicator_id, value, quality_score)
                VALUES (?, ?, ?, ?)
            ''', [row['date'], ind_id, row['value'], row.get('quality_score', 1.0)])
        
        conn.close()
        
        time.sleep(delay)  # Rate limiting
        return True, f"{len(df_clean)} rows"
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Bulk fetch indicators")
    parser.add_argument("--registry", default="prism/registry/indicators_expanded.yaml")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--resume", action="store_true", help="Skip already fetched indicators")
    parser.add_argument("--source", choices=['fred', 'tiingo', 'all'], default='all')
    parser.add_argument("--category", help="Only fetch specific category")
    parser.add_argument("--limit", type=int, help="Max indicators to fetch")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PRISM BULK FETCHER")
    print("=" * 60)
    
    # Load registry
    indicators = load_expanded_registry(args.registry)
    print(f"Loaded {len(indicators)} indicators from registry")
    
    # Filter by source
    if args.source != 'all':
        indicators = [i for i in indicators if i['source'] == args.source]
        print(f"Filtered to {len(indicators)} {args.source} indicators")
    
    # Filter by category
    if args.category:
        indicators = [i for i in indicators if i.get('category') == args.category]
        print(f"Filtered to {len(indicators)} {args.category} indicators")
    
    # Resume mode
    if args.resume:
        fetched = get_fetched_indicators()
        indicators = [i for i in indicators if i['id'] not in fetched]
        print(f"Resume mode: {len(indicators)} remaining to fetch")
    
    # Limit
    if args.limit:
        indicators = indicators[:args.limit]
        print(f"Limited to {len(indicators)} indicators")
    
    if not indicators:
        print("No indicators to fetch!")
        return
    
    # Fetch
    print(f"\nFetching {len(indicators)} indicators (delay={args.delay}s)...")
    print("-" * 60)
    
    succeeded = 0
    failed = 0
    start_time = time.time()
    
    for i, indicator in enumerate(indicators):
        ind_id = indicator['id']
        source = indicator['source']
        
        success, msg = fetch_single_indicator(indicator, args.delay)
        
        status = "✓" if success else "✗"
        print(f"[{i+1}/{len(indicators)}] {status} {ind_id:<20} ({source}) - {msg}")
        
        if success:
            succeeded += 1
        else:
            failed += 1
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:     {len(indicators)}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed:    {failed}")
    print(f"Time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
