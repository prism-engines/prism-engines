#!/usr/bin/env python3
"""
PRISM Quarterly Regime Analysis
================================
Pull data from database, create panel, run regime detection over 30 years.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

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


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Database path
DB_PATH = get_db_path()

def load_from_database():
    """Load all data from database using unified indicator_values table."""
    from data.sql.db_connector import load_all_indicators_wide

    panel = load_all_indicators_wide()

    if panel.empty:
        import warnings
        warnings.warn(
            "indicator_values table is empty. Data may need migration.",
            DeprecationWarning
        )
        return pd.DataFrame()

    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    return panel


def resample_quarterly(df):
    """Resample daily data to quarterly (end of quarter values)."""
    return df.resample('Q').last()


def run_regime_analysis(panel, n_regimes=3):
    """Run regime switching analysis on the panel."""
    from engine_core.lenses.regime_switching_lens import RegimeSwitchingLens
    
    lens = RegimeSwitchingLens()
    
    # Drop columns with too many NaN
    valid_cols = panel.columns[panel.notna().sum() > len(panel) * 0.5]
    clean_panel = panel[valid_cols].copy()
    
    # Forward fill then drop remaining NaN rows
    clean_panel = clean_panel.ffill().dropna()
    
    if len(clean_panel) < 10:
        print(f"ERROR: Not enough data points ({len(clean_panel)})")
        return None
    
    print(f"Running regime analysis on {len(clean_panel)} periods, {len(valid_cols)} indicators")
    
    # Reset index to make date a column
    clean_panel = clean_panel.reset_index()
    clean_panel = clean_panel.rename(columns={'index': 'date'})
    
    results = lens.analyze(clean_panel, n_regimes=n_regimes)
    
    return results, clean_panel


def print_regime_summary(results, panel):
    """Print a summary of detected regimes."""
    print("\n" + "=" * 70)
    print("PRISM REGIME DETECTION RESULTS")
    print("=" * 70)
    
    if 'regime_stats' in results:
        print("\n📊 REGIME STATISTICS")
        print("-" * 50)
        for regime_id, stats in results['regime_stats'].items():
            print(f"\nRegime {regime_id}:")
            print(f"  Periods: {stats.get('count', 'N/A')} ({stats.get('pct', 0):.1f}%)")
            if 'mean_values' in stats:
                print(f"  Key characteristics:")
                # Show top 5 distinguishing features
                means = stats['mean_values']
                sorted_means = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                for col, val in sorted_means:
                    print(f"    {col}: {val:.3f}")
    
    if 'regime_labels' in results:
        labels = results['regime_labels']
        print(f"\n📅 REGIME TIMELINE")
        print("-" * 50)
        
        # Show regime by date
        if 'date' in panel.columns:
            dates = panel['date'].values
            for i, (date, regime) in enumerate(zip(dates, labels)):
                if i % 4 == 0:  # Print every 4th quarter (yearly)
                    print(f"  {str(date)[:10]}: Regime {regime}")
    
    if 'transitions' in results:
        print(f"\n🔄 REGIME TRANSITIONS")
        print("-" * 50)
        trans = results['transitions']
        print(f"  Total transitions: {trans.get('total', 'N/A')}")
        if 'matrix' in trans:
            print(f"  Transition matrix:")
            print(trans['matrix'])
    
    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("PRISM QUARTERLY REGIME ANALYSIS - 30 YEARS")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data from database...")
    panel = load_from_database()
    print(f"  Loaded {len(panel)} daily observations")
    print(f"  Date range: {panel.index.min()} to {panel.index.max()}")
    print(f"  Indicators: {len(panel.columns)}")
    
    # Filter to last 30 years
    cutoff = panel.index.max() - pd.DateOffset(years=50)
    panel_30y = panel[panel.index >= cutoff]
    print(f"\n📅 Filtering to last 30 years...")
    print(f"  From: {panel_30y.index.min()}")
    print(f"  To: {panel_30y.index.max()}")
    print(f"  Daily observations: {len(panel_30y)}")
    
    # Resample to quarterly
    print("\n📊 Resampling to quarterly...")
    quarterly = resample_quarterly(panel_30y)
    print(f"  Quarterly observations: {len(quarterly)}")
    
    # Run regime analysis
    print("\n🔬 Running regime detection...")
    results, clean_panel = run_regime_analysis(quarterly, n_regimes=3)
    
    if results:
        print_regime_summary(results, clean_panel)
        
        # Save results
        output_dir = OUTPUT_DIR / "regime_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save regime labels with dates
        if 'regime_labels' in results and 'date' in clean_panel.columns:
            regime_df = pd.DataFrame({
                'date': clean_panel['date'],
                'regime': results['regime_labels']
            })
            regime_df.to_csv(output_dir / "quarterly_regimes_30y.csv", index=False)
            print(f"\n💾 Saved to {output_dir / 'quarterly_regimes_30y.csv'}")
    
    return results


if __name__ == "__main__":
    main()