#!/usr/bin/env python3
"""
Query PRISM Analysis Results
============================
Simple CLI to inspect past analysis runs stored in the database.

Usage:
    python query_results.py                    # Show recent runs
    python query_results.py --run 5            # Show details for run 5
    python query_results.py --indicator uso    # Show USO ranking history
    python query_results.py --compare 5 6      # Compare two runs
"""

import sys
import argparse
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'start' else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.duckdb_connector import (
    get_analysis_runs,
    get_run_results, 
    get_indicator_rankings,
    query
)


def show_recent_runs(limit: int = 10):
    """Show recent analysis runs."""
    df = get_analysis_runs(limit=limit)
    
    if df.empty:
        print("No analysis runs found in database.")
        return
    
    print("\n📊 Recent Analysis Runs")
    print("=" * 80)
    print(f"{'ID':<5} {'Date':<20} {'Range':<25} {'Ind':<5} {'Lens':<5} {'Err':<4}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        date_range = f"{row['start_date']} → {row['end_date'] or 'today'}"
        print(f"{row['run_id']:<5} {str(row['run_date'])[:19]:<20} {date_range:<25} "
              f"{row['n_indicators']:<5} {row['n_lenses']:<5} {row['n_errors']:<4}")


def show_run_details(run_id: int):
    """Show details for a specific run."""
    result = get_run_results(run_id)
    
    if not result:
        print(f"Run {run_id} not found.")
        return
    
    print(f"\n📊 Run {run_id} Details")
    print("=" * 60)
    print(f"Date:       {result.get('run_date')}")
    print(f"Range:      {result.get('start_date')} → {result.get('end_date') or 'today'}")
    print(f"Indicators: {result.get('n_indicators')}")
    print(f"Rows:       {result.get('n_rows')}")
    print(f"Lenses:     {result.get('n_lenses')}")
    print(f"Errors:     {result.get('n_errors')}")
    
    if result.get('lens_errors'):
        print("\n⚠️  Errors:")
        for lens, error in result['lens_errors'].items():
            print(f"   {lens}: {error[:60]}...")
    
    consensus = result.get('consensus')
    if consensus is not None and not consensus.empty:
        print("\n🏆 Top 10 Indicators (Consensus)")
        print("-" * 40)
        for _, row in consensus.head(10).iterrows():
            print(f"   {row['rank']:>2}. {row['indicator']:<20} {row['mean_score']:.3f} ({row['n_lenses']} lenses)")


def show_indicator_history(indicator: str, limit: int = 10):
    """Show ranking history for an indicator."""
    df = get_indicator_rankings(indicator, limit=limit)
    
    if df.empty:
        print(f"No rankings found for '{indicator}'")
        return
    
    print(f"\n📈 Ranking History: {indicator}")
    print("=" * 70)
    print(f"{'Run':<5} {'Date':<20} {'Range':<20} {'Score':<8} {'Rank':<5}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        date_range = f"{row['start_date'][:10]}"
        print(f"{row['run_id']:<5} {str(row['run_date'])[:19]:<20} {date_range:<20} "
              f"{row['mean_score']:<8.3f} #{row['rank']:<4}")


def compare_runs(run_id1: int, run_id2: int):
    """Compare top indicators between two runs."""
    r1 = get_run_results(run_id1)
    r2 = get_run_results(run_id2)
    
    if not r1 or not r2:
        print("One or both runs not found.")
        return
    
    c1 = r1.get('consensus')
    c2 = r2.get('consensus')
    
    if c1 is None or c2 is None or c1.empty or c2.empty:
        print("Missing consensus data.")
        return
    
    print(f"\n🔄 Comparison: Run {run_id1} vs Run {run_id2}")
    print("=" * 60)
    
    # Create lookup dicts
    ranks1 = dict(zip(c1['indicator'], c1['rank']))
    ranks2 = dict(zip(c2['indicator'], c2['rank']))
    
    all_indicators = set(ranks1.keys()) | set(ranks2.keys())
    
    # Find biggest movers
    changes = []
    for ind in all_indicators:
        r1_rank = ranks1.get(ind, 99)
        r2_rank = ranks2.get(ind, 99)
        change = r1_rank - r2_rank  # Positive = improved in run2
        changes.append((ind, r1_rank, r2_rank, change))
    
    changes.sort(key=lambda x: x[3], reverse=True)
    
    print("\n📈 Biggest Gainers (improved rank):")
    for ind, r1, r2, chg in changes[:5]:
        if chg > 0:
            print(f"   {ind}: #{r1} → #{r2} (+{chg})")
    
    print("\n📉 Biggest Losers (dropped rank):")
    for ind, r1, r2, chg in changes[-5:]:
        if chg < 0:
            print(f"   {ind}: #{r1} → #{r2} ({chg})")


def main():
    parser = argparse.ArgumentParser(description='Query PRISM Analysis Results')
    parser.add_argument('--run', '-r', type=int, help='Show details for specific run ID')
    parser.add_argument('--indicator', '-i', type=str, help='Show ranking history for indicator')
    parser.add_argument('--compare', '-c', nargs=2, type=int, help='Compare two run IDs')
    parser.add_argument('--limit', '-n', type=int, default=10, help='Number of results to show')
    
    args = parser.parse_args()
    
    if args.run:
        show_run_details(args.run)
    elif args.indicator:
        show_indicator_history(args.indicator, limit=args.limit)
    elif args.compare:
        compare_runs(args.compare[0], args.compare[1])
    else:
        show_recent_runs(limit=args.limit)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
