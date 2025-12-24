#!/usr/bin/env python3
"""
PRISM Multi-View Geometry Runner

Runs multi-view geometry analysis on real indicators from the database.

Usage:
    python scripts/run_multiview_geometry.py
    python scripts/run_multiview_geometry.py --indicators SPY,GLD,VIXCLS
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd

from prism.db.connection import get_connection


def main():
    parser = argparse.ArgumentParser(description="Multi-view geometry analysis")
    parser.add_argument("--indicators", type=str, help="Comma-separated indicator list")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum rows required")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Import after path setup
    from agent_multiview_geometry import (
        MultiViewGeometryAgent,
        ArbitrationAgent,
        ViewType,
    )
    from scripts.agent_geometry_signature import GeometrySignatureAgent
    
    # Load data
    conn = get_connection()
    df = conn.execute("""
        SELECT indicator_id, date, value
        FROM clean.indicator_values
        ORDER BY indicator_id, date
    """).fetchdf()
    conn.close()
    
    # Determine indicators to analyze
    if args.indicators:
        indicator_list = [x.strip() for x in args.indicators.split(",")]
    else:
        # Get all indicators with enough data
        counts = df.groupby('indicator_id').size()
        indicator_list = counts[counts >= args.min_rows].index.tolist()
    
    print("=" * 70)
    print(f"Multi-View Geometry Analysis - {len(indicator_list)} indicators")
    print("=" * 70)
    
    # Initialize agents
    base_agent = GeometrySignatureAgent(verbose=False)
    mv_agent = MultiViewGeometryAgent(base_agent=base_agent, verbose=args.verbose)
    arbitration = ArbitrationAgent(verbose=args.verbose)
    
    # Spreads get different views
    spread_indicators = {'T10Y2Y', 'T10Y3M', 'T10YFF', 'TEDRATE'}
    
    results = []
    
    for ind_id in indicator_list:
        ind_data = df[df['indicator_id'] == ind_id]['value'].dropna().values
        
        if len(ind_data) < args.min_rows:
            continue
        
        # Use last 1000 points
        series = ind_data[-1000:] if len(ind_data) > 1000 else ind_data
        
        try:
            # Choose views based on indicator type
            if ind_id in spread_indicators:
                views = [ViewType.LEVEL, ViewType.DEVIATION]
            else:
                views = [ViewType.LEVEL, ViewType.RETURNS, ViewType.VOLATILITY]
            
            result = mv_agent.analyze(series, indicator_id=ind_id, views=views)
            results.append(result)
            
            # Print individual result
            if not args.verbose:
                level_geom = result.views.get(ViewType.LEVEL)
                returns_geom = result.views.get(ViewType.RETURNS)
                vol_geom = result.views.get(ViewType.VOLATILITY)
                dev_geom = result.views.get(ViewType.DEVIATION)
                
                parts = [f"{ind_id:12}"]
                if level_geom:
                    parts.append(f"L:{level_geom.dominant_geometry[:6]:6}({level_geom.confidence:.2f})")
                if returns_geom:
                    parts.append(f"R:{returns_geom.dominant_geometry[:6]:6}({returns_geom.confidence:.2f})")
                if vol_geom:
                    parts.append(f"V:{vol_geom.dominant_geometry[:6]:6}({vol_geom.confidence:.2f})")
                if dev_geom:
                    parts.append(f"D:{dev_geom.dominant_geometry[:6]:6}({dev_geom.confidence:.2f})")
                parts.append(f"→ {result.consensus_geometry}")
                
                print(" | ".join(parts))
            
        except Exception as e:
            print(f"Error on {ind_id}: {e}")
    
    # Arbitration report
    arbitration.results = results
    print("\n" + arbitration.generate_report())
    
    # Show some example detailed results
    print("\n" + "=" * 70)
    print("EXAMPLE DETAILED RESULTS")
    print("=" * 70)
    
    # Find interesting cases
    examples = {
        "SPY": "Equity (expect level=latent, returns=reflexive)",
        "VIXCLS": "Volatility index",
        "DGS10": "Treasury yield",
        "T10Y2Y": "Yield spread (expect oscillator in deviation)",
        "CPIAUCSL": "CPI (trend)",
        "GDP": "GDP (quarterly)",
    }
    
    for ind_id, desc in examples.items():
        result = next((r for r in results if r.indicator_id == ind_id), None)
        if result:
            print(f"\n{desc}:")
            print(result.summary())
    
    # Disagreement analysis
    print("\n" + arbitration.explain_disagreements())
    
    # Summary table by geometry
    print("\n" + "=" * 70)
    print("BY CONSENSUS GEOMETRY")
    print("=" * 70)
    
    by_geometry = {}
    for r in results:
        geom = r.consensus_geometry
        if geom not in by_geometry:
            by_geometry[geom] = []
        by_geometry[geom].append(r)
    
    for geom in sorted(by_geometry.keys()):
        items = by_geometry[geom]
        print(f"\n{geom.upper()} ({len(items)} indicators):")
        # Show top 10 by confidence
        sorted_items = sorted(items, key=lambda x: -x.consensus_confidence)[:10]
        for r in sorted_items:
            print(f"  {r.indicator_id:12} conf={r.consensus_confidence:.2f} disagree={r.disagreement_score:.2f}")


if __name__ == "__main__":
    main()
