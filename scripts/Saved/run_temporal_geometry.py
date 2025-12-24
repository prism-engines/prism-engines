#!/usr/bin/env python3
"""
PRISM Temporal Geometry Analysis

Minimal script to track geometry signature evolution over time.

Uses a rolling window (default 7 years) to compute geometry at each point,
revealing when market structure transitions between regimes.

Example insight:
    - SPY 2010-2017: latent_flow (calm accumulation)
    - SPY 2018 Q4:   reflexive_stochastic (vol spike)
    - SPY 2020 Mar:  reflexive_stochastic (COVID crash)
    - SPY 2021:      latent_flow (recovery trend)

Usage:
    python scripts/run_temporal_geometry.py --indicator SPY
    python scripts/run_temporal_geometry.py --indicator SPY --window-years 5
    python scripts/run_temporal_geometry.py --indicator SPY,QQQ,DGS10 --step-days 21

Cross-validated by: Claude, GPT-4
Date: December 2024
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from prism.db.connection import get_connection


def run_temporal_geometry(
    indicator_id: str,
    window_years: float = 7.0,
    step_days: int = 63,  # Quarterly steps by default
    min_window_pct: float = 0.8,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run rolling-window geometry analysis on a single indicator.
    
    Args:
        indicator_id: Indicator to analyze
        window_years: Rolling window size in years
        step_days: Days between analysis points
        min_window_pct: Minimum data coverage required
        verbose: Print progress
        
    Returns:
        DataFrame with columns: date, geometry, confidence, disagreement, 
                                level_geom, returns_geom, vol_geom
    """
    from scripts.agent_multiview_geometry import (
        MultiViewGeometryAgent,
        ViewType,
    )
    from scripts.agent_geometry_signature import GeometrySignatureAgent
    
    # Load data
    conn = get_connection()
    df = conn.execute("""
        SELECT date, value
        FROM clean.indicator_values
        WHERE indicator_id = ?
        ORDER BY date
    """, [indicator_id]).fetchdf()
    conn.close()
    
    if len(df) < 252:  # Need at least 1 year
        raise ValueError(f"{indicator_id}: insufficient data ({len(df)} rows)")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Calculate window size in trading days (~252/year)
    window_days = int(window_years * 252)
    min_required = int(window_days * min_window_pct)
    
    # Initialize agents
    base_agent = GeometrySignatureAgent(verbose=False)
    mv_agent = MultiViewGeometryAgent(base_agent=base_agent, verbose=False)
    
    # Determine if this is a spread indicator
    spread_indicators = {'T10Y2Y', 'T10Y3M', 'T10YFF', 'TEDRATE'}
    is_spread = indicator_id in spread_indicators
    
    # Rolling analysis
    results = []
    dates = df.index.tolist()
    
    # Start after we have enough data for first window
    start_idx = window_days
    
    if verbose:
        print(f"{indicator_id}: {len(dates)} days, window={window_days} days, step={step_days}")
        print(f"  Analyzing from {dates[start_idx].date()} to {dates[-1].date()}")
    
    for end_idx in range(start_idx, len(dates), step_days):
        start_idx_window = end_idx - window_days
        window_data = df.iloc[start_idx_window:end_idx]['value'].dropna()
        
        if len(window_data) < min_required:
            continue
        
        series = window_data.values
        end_date = dates[end_idx]
        
        try:
            if is_spread:
                views = [ViewType.LEVEL, ViewType.DEVIATION]
            else:
                views = [ViewType.LEVEL, ViewType.RETURNS, ViewType.VOLATILITY]
            
            result = mv_agent.analyze(series, indicator_id=indicator_id, views=views)
            
            # Extract view-level results
            level_geom = None
            returns_geom = None
            vol_geom = None
            dev_geom = None
            
            for vt, vr in result.views.items():
                if vt == ViewType.LEVEL:
                    level_geom = vr.dominant_geometry
                elif vt == ViewType.RETURNS:
                    returns_geom = vr.dominant_geometry
                elif vt == ViewType.VOLATILITY:
                    vol_geom = vr.dominant_geometry
                elif vt == ViewType.DEVIATION:
                    dev_geom = vr.dominant_geometry
            
            results.append({
                'date': end_date,
                'geometry': result.consensus_geometry,
                'confidence': result.consensus_confidence,
                'disagreement': result.disagreement_score,
                'level_geom': level_geom,
                'returns_geom': returns_geom,
                'vol_geom': vol_geom,
                'dev_geom': dev_geom,
                'n_samples': len(window_data),
            })
            
            if verbose and len(results) % 10 == 0:
                print(f"  {end_date.date()}: {result.consensus_geometry} "
                      f"(conf={result.consensus_confidence:.2f})")
                
        except Exception as e:
            if verbose:
                print(f"  {end_date.date()}: ERROR - {e}")
    
    return pd.DataFrame(results)


def find_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find geometry transition points.
    
    Returns DataFrame with transition dates and before/after geometry.
    """
    if len(df) < 2:
        return pd.DataFrame()
    
    transitions = []
    prev_geom = df.iloc[0]['geometry']
    
    for i in range(1, len(df)):
        curr_geom = df.iloc[i]['geometry']
        if curr_geom != prev_geom:
            transitions.append({
                'date': df.iloc[i]['date'],
                'from_geometry': prev_geom,
                'to_geometry': curr_geom,
                'confidence': df.iloc[i]['confidence'],
            })
        prev_geom = curr_geom
    
    return pd.DataFrame(transitions)


def summarize_temporal(df: pd.DataFrame, indicator_id: str) -> None:
    """Print summary of temporal geometry evolution."""
    print(f"\n{'='*60}")
    print(f"TEMPORAL GEOMETRY: {indicator_id}")
    print(f"{'='*60}")
    print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Observations: {len(df)}")
    
    # Geometry distribution over time
    print(f"\nGeometry Distribution:")
    geom_counts = df['geometry'].value_counts()
    for geom, count in geom_counts.items():
        pct = 100 * count / len(df)
        print(f"  {geom:24} {count:4} ({pct:5.1f}%)")
    
    # Transitions
    transitions = find_transitions(df)
    print(f"\nTransitions: {len(transitions)}")
    if len(transitions) > 0:
        for _, t in transitions.iterrows():
            print(f"  {t['date'].date()}: {t['from_geometry']} → {t['to_geometry']}")
    
    # Recent state
    recent = df.iloc[-5:]
    print(f"\nRecent (last 5 observations):")
    for _, row in recent.iterrows():
        print(f"  {row['date'].date()}: {row['geometry']:24} "
              f"(conf={row['confidence']:.2f}, disagree={row['disagreement']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Temporal Geometry Analysis")
    parser.add_argument("--indicator", "-i", type=str, required=True,
                       help="Indicator ID(s), comma-separated")
    parser.add_argument("--window-years", "-w", type=float, default=7.0,
                       help="Rolling window in years (default: 7)")
    parser.add_argument("--step-days", "-s", type=int, default=63,
                       help="Days between observations (default: 63 = quarterly)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output CSV path (optional)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    args = parser.parse_args()
    
    indicators = [x.strip() for x in args.indicator.split(",")]
    
    print("=" * 60)
    print("PRISM Temporal Geometry Analysis")
    print(f"Window: {args.window_years} years, Step: {args.step_days} days")
    print("=" * 60)
    
    all_results = []
    
    for ind in indicators:
        try:
            df = run_temporal_geometry(
                indicator_id=ind,
                window_years=args.window_years,
                step_days=args.step_days,
                verbose=args.verbose,
            )
            df['indicator_id'] = ind
            all_results.append(df)
            
            summarize_temporal(df, ind)
            
        except Exception as e:
            print(f"\n{ind}: ERROR - {e}")
    
    if all_results and args.output:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
