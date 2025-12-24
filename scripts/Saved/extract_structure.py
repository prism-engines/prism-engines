#!/usr/bin/env python3
"""
PRISM Structure Analysis

Extract geometry, identify states, run bounded analysis.

Usage:
    python scripts/extract_structure.py
    python scripts/extract_structure.py --states 4 --window 126
    python scripts/extract_structure.py --output results/structure.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.db.connection import get_db_path, get_connection
from prism.structure import StructureExtractor


def load_data(db_path: str, indicators: list, start_date: str = None) -> pd.DataFrame:
    """Load indicator data."""
    conn = get_connection(Path(db_path) if db_path else None)
    ind_list = ", ".join(f"'{i}'" for i in indicators)
    
    query = f"""
        SELECT date, indicator_id, value
        FROM clean.indicators
        WHERE indicator_id IN ({ind_list})
    """
    if start_date:
        query += f" AND date >= '{start_date}'"
    
    df = conn.execute(query).fetchdf()
    df_wide = df.pivot(index='date', columns='indicator_id', values='value').dropna()
    return df_wide


def main():
    parser = argparse.ArgumentParser(description="Extract PRISM structure")
    parser.add_argument("--db", default=str(get_db_path()))
    parser.add_argument("--indicators", default="SPY,XLU,XLF,XLK,XLE,AGG,GLD,VIXCLS")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--window", type=int, default=252, help="Rolling window size")
    parser.add_argument("--step", type=int, default=21, help="Step between windows")
    parser.add_argument("--states", type=int, default=3, help="Number of states")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    indicators = [i.strip() for i in args.indicators.split(",")]
    
    print("=" * 60)
    print("PRISM STRUCTURE EXTRACTION")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data...")
    df = load_data(args.db, indicators, args.start)
    print(f"   {len(df)} rows, {len(df.columns)} indicators")
    print(f"   Period: {df.index.min().date()} to {df.index.max().date()}")
    
    # Extract geometry
    print(f"\n2. Extracting geometry (window={args.window}, step={args.step})...")
    extractor = StructureExtractor(df)
    geometry = extractor.extract_geometry(
        window=args.window,
        step=args.step,
        verbose=True
    )
    
    # Cluster into states
    print(f"\n3. Clustering geometry into {args.states} states...")
    state_labels = extractor.cluster_geometry(n_states=args.states)
    
    # Print state summary
    print()
    extractor.print_state_summary()
    
    # Run bounded analysis
    print("\n" + "=" * 60)
    print("BOUNDED ANALYSIS")
    print("=" * 60)
    
    engines_to_compare = ["pca", "hurst", "cross_correlation", "entropy"]
    bounded_results = extractor.run_all_bounded(engines=engines_to_compare)
    
    # Print comparison
    print("\nMetric comparison across states:")
    print("-" * 60)
    
    metrics_to_show = [
        ("pca", "variance_pc1"),
        ("pca", "effective_dimensionality"),
        ("hurst", "avg_hurst"),
        ("cross_correlation", "avg_abs_correlation"),
        ("entropy", "avg_permutation_entropy"),
    ]
    
    # Header
    header = f"{'Metric':<40}"
    for state_id in sorted(extractor.states.keys()):
        header += f" State {state_id:>6}"
    print(header)
    print("-" * 60)
    
    # Rows
    for engine_name, metric_name in metrics_to_show:
        row = f"{engine_name}/{metric_name:<30}"
        for state_id in sorted(extractor.states.keys()):
            if engine_name in bounded_results[state_id]:
                val = bounded_results[state_id][engine_name].get(metric_name, None)
                if val is not None:
                    row += f" {val:>8.4f}"
                else:
                    row += f" {'N/A':>8}"
            else:
                row += f" {'ERR':>8}"
        print(row)
    
    # Save results
    if args.output:
        results = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "indicators": indicators,
                "window": args.window,
                "step": args.step,
                "n_states": args.states,
                "date_range": {
                    "start": str(df.index.min().date()),
                    "end": str(df.index.max().date()),
                }
            },
            "geometry": {
                "n_points": len(geometry),
                "dimensions": list(geometry.columns),
            },
            "states": {},
            "bounded_analysis": {},
        }
        
        for state_id, state in extractor.states.items():
            results["states"][state_id] = {
                "n_points": state.n_points,
                "centroid": {k: round(v, 6) for k, v in state.centroid.items()},
                "date_ranges": [(str(s), str(e)) for s, e in state.date_ranges],
            }
        
        for state_id, engines in bounded_results.items():
            results["bounded_analysis"][state_id] = {}
            for eng_name, metrics in engines.items():
                results["bounded_analysis"][state_id][eng_name] = {
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in metrics.items()
                }
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("Structure extraction complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
