#!/usr/bin/env python3
"""
PRISM Deformation Analysis

1. How early did geometry deform before 2008?
2. Any recent deformation?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from prism.db.connection import get_db_path, get_connection
from prism.structure import StructureExtractor


def load_data(db_path=None):
    conn = get_connection(Path(db_path) if db_path else None)
    df = conn.execute('''
        SELECT date, indicator_id, value
        FROM clean.indicators
        WHERE indicator_id IN ('SPY', 'XLU', 'XLF', 'XLK', 'XLE', 'AGG', 'GLD', 'VIXCLS')
        AND date >= '2005-01-01'
    ''').fetchdf()
    return df.pivot(index='date', columns='indicator_id', values='value').dropna()


def main():
    print("=" * 70)
    print("PRISM DEFORMATION TIMELINE ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    df_wide = load_data()
    print(f"   {len(df_wide)} rows: {df_wide.index.min().date()} to {df_wide.index.max().date()}")
    
    # Extract geometry with fine resolution
    print("\n2. Extracting geometry (fine resolution)...")
    extractor = StructureExtractor(df_wide)
    
    geometry = extractor.extract_geometry(
        window=126,  # 6 months
        step=5,      # Weekly resolution
        verbose=False
    )
    
    state_labels = extractor.cluster_geometry(n_states=4)
    print(f"   {len(geometry)} geometry points")
    
    # Get geometry DataFrame with states
    geo_df = geometry.copy()
    geo_df['state'] = state_labels
    
    # Convert index to datetime if it's date objects
    geo_df.index = pd.to_datetime(geo_df.index)
    
    # Identify the "stress" state (highest effective dimensionality / lowest PC1)
    state_chars = {}
    for state_id in state_labels.unique():
        mask = state_labels == state_id
        state_geo = geometry[mask]
        state_chars[state_id] = {
            'pc1': state_geo['pca__variance_pc1'].mean(),
            'eff_dim': state_geo['pca__effective_dimensionality'].mean(),
            'corr': state_geo['cross_correlation__avg_abs_correlation'].mean(),
            'count': mask.sum()
        }
    
    # Find stress state (lowest PC1 or highest eff_dim)
    stress_state = min(state_chars.keys(), key=lambda s: state_chars[s]['pc1'])
    
    print(f"\n3. State Characteristics:")
    print("-" * 50)
    for state_id, chars in sorted(state_chars.items()):
        marker = " ← STRESS STATE" if state_id == stress_state else ""
        print(f"   State {state_id}: PC1={chars['pc1']:.3f}, eff_dim={chars['eff_dim']:.2f}, "
              f"corr={chars['corr']:.3f}, n={chars['count']}{marker}")
    
    # =========================================================================
    # PRE-2008 ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PRE-2008 DEFORMATION TIMELINE")
    print("=" * 70)
    
    # Lehman was Sept 15, 2008
    lehman_date = pd.Timestamp('2008-09-15')
    
    # Find first stress state occurrence before Lehman
    pre_lehman = geo_df[geo_df.index < lehman_date]
    stress_periods = pre_lehman[pre_lehman['state'] == stress_state]
    
    if len(stress_periods) > 0:
        first_stress = stress_periods.index.min()
        days_before = (lehman_date - first_stress).days
        
        print(f"\nStress state (State {stress_state}) FIRST appeared: {first_stress.date()}")
        print(f"Lehman Brothers collapse: {lehman_date.date()}")
        print(f"WARNING LEAD TIME: {days_before} days ({days_before/30:.1f} months)")
        
        # Timeline of state transitions pre-2008
        print(f"\nPre-crisis state timeline:")
        print("-" * 50)
        
        pre_crisis = geo_df[(geo_df.index >= '2007-01-01') & (geo_df.index <= '2008-10-01')]
        
        current_state = None
        transitions = []
        for date, row in pre_crisis.iterrows():
            if row['state'] != current_state:
                transitions.append((date, int(row['state'])))
                current_state = row['state']
        
        for date, state in transitions:
            marker = " *** STRESS" if state == stress_state else ""
            days_to_lehman = (lehman_date - date).days
            print(f"   {date.date()} → State {state} ({days_to_lehman:>4} days before Lehman){marker}")
    else:
        print("\nNo stress state detected before Lehman")
    
    # =========================================================================
    # RECENT ANALYSIS (2023-2025)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RECENT DEFORMATION ANALYSIS (2023-2025)")
    print("=" * 70)
    
    recent = geo_df[geo_df.index >= '2023-01-01']
    
    print(f"\nRecent state distribution:")
    print("-" * 50)
    for state_id in sorted(state_labels.unique()):
        count = (recent['state'] == state_id).sum()
        pct = count / len(recent) * 100 if len(recent) > 0 else 0
        marker = " ← STRESS STATE" if state_id == stress_state else ""
        print(f"   State {state_id}: {count:>3} points ({pct:>5.1f}%){marker}")
    
    # Recent timeline
    print(f"\nRecent state transitions:")
    print("-" * 50)
    
    current_state = None
    for date, row in recent.iterrows():
        if row['state'] != current_state:
            marker = " *** STRESS" if row['state'] == stress_state else ""
            print(f"   {date.date()} → State {int(row['state'])}{marker}")
            current_state = row['state']
    
    # Check last 6 months specifically
    print(f"\nLast 6 months:")
    print("-" * 50)
    
    last_6mo = geo_df[geo_df.index >= (geo_df.index.max() - pd.Timedelta(days=180))]
    
    stress_count = (last_6mo['state'] == stress_state).sum()
    stress_pct = stress_count / len(last_6mo) * 100 if len(last_6mo) > 0 else 0
    
    print(f"   Total geometry points: {len(last_6mo)}")
    print(f"   In stress state: {stress_count} ({stress_pct:.1f}%)")
    
    # Current state
    current = geo_df.iloc[-1]
    print(f"\n   CURRENT STATE (as of {geo_df.index[-1].date()}): State {int(current['state'])}")
    
    if current['state'] == stress_state:
        print(f"   ⚠️  CURRENTLY IN STRESS STATE")
    else:
        print(f"   ✓  Not in stress state")
    
    # Show current geometry values
    print(f"\n   Current geometry:")
    for col in ['pca__variance_pc1', 'pca__effective_dimensionality', 
                'cross_correlation__avg_abs_correlation', 'lyapunov__avg_lyapunov']:
        if col in current.index:
            print(f"      {col.split('__')[1]}: {current[col]:.4f}")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Pre-2008 vs Now")
    print("=" * 70)
    
    # Get geometry values for different periods
    pre_2008_stress = geo_df[(geo_df.index >= '2008-01-01') & (geo_df.index <= '2008-09-01')]
    
    print(f"\nGeometry comparison:")
    print("-" * 60)
    print(f"{'Metric':<30} {'Pre-2008 Avg':>15} {'Current':>15}")
    print("-" * 60)
    
    for col in ['pca__variance_pc1', 'pca__effective_dimensionality', 
                'cross_correlation__avg_abs_correlation']:
        if col in pre_2008_stress.columns and col in current.index:
            pre_val = pre_2008_stress[col].mean()
            cur_val = current[col]
            print(f"{col.split('__')[1]:<30} {pre_val:>15.4f} {cur_val:>15.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
