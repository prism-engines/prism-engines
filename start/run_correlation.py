#!/usr/bin/env python3
"""
PRISM Correlation Matrix Analysis
==================================
Generate correlation matrix for all indicators.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

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


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Database path
DB_PATH = DATA_DIR / "prism.db"

def load_from_database():
    """Load all data from database and merge into a single panel."""
    conn = sqlite3.connect(DB_PATH)
    
    # Load market data
    market_df = pd.read_sql("""
        SELECT date, ticker, value 
        FROM market_prices 
        ORDER BY date
    """, conn)
    
    if not market_df.empty:
        market_wide = market_df.pivot(index='date', columns='ticker', values='value')
    else:
        market_wide = pd.DataFrame()
    
    # Load economic data
    econ_df = pd.read_sql("""
        SELECT date, series_id, value 
        FROM econ_values 
        ORDER BY date
    """, conn)
    
    if not econ_df.empty:
        econ_wide = econ_df.pivot(index='date', columns='series_id', values='value')
    else:
        econ_wide = pd.DataFrame()
    
    conn.close()
    
    # Merge on date
    if not market_wide.empty and not econ_wide.empty:
        panel = market_wide.join(econ_wide, how='outer')
    elif not market_wide.empty:
        panel = market_wide
    else:
        panel = econ_wide
    
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    
    return panel


def compute_returns(df):
    """Compute daily returns (percent change)."""
    return df.pct_change().dropna()


def main():
    print("=" * 70)
    print("PRISM CORRELATION MATRIX ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data from database...")
    panel = load_from_database()
    print(f"  Loaded {len(panel)} observations, {len(panel.columns)} indicators")
    
    # Filter to recent 10 years for better coverage
    cutoff = panel.index.max() - pd.DateOffset(years=10)
    panel_recent = panel[panel.index >= cutoff]
    print(f"\n📅 Using last 10 years: {panel_recent.index.min().date()} to {panel_recent.index.max().date()}")
    
    # Drop columns with too many NaN (50% threshold)
    valid_cols = panel_recent.columns[panel_recent.notna().sum() > len(panel_recent) * 0.5]
    clean_panel = panel_recent[valid_cols].ffill().bfill().dropna()
    print(f"  Valid indicators: {len(valid_cols)}")
    
    # Compute returns
    print("\n📊 Computing returns...")
    returns = compute_returns(clean_panel)
    print(f"  Return observations: {len(returns)}")
    
    # Compute correlation matrix
    print("\n🔬 Computing correlation matrix...")
    corr_matrix = returns.corr()
    
    # Print correlation matrix as text
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX (Top correlations)")
    print("=" * 70)
    
    # Get top positive and negative correlations
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                corr_pairs.append((col1, col2, corr_val))
    
    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\n🔥 STRONGEST POSITIVE CORRELATIONS")
    print("-" * 50)
    positive = [p for p in corr_pairs if p[2] > 0][:15]
    for col1, col2, corr in positive:
        print(f"  {col1:20} <-> {col2:20}: {corr:+.3f}")
    
    print("\n❄️  STRONGEST NEGATIVE CORRELATIONS")
    print("-" * 50)
    negative = [p for p in corr_pairs if p[2] < 0][:15]
    for col1, col2, corr in negative:
        print(f"  {col1:20} <-> {col2:20}: {corr:+.3f}")
    
    # Group analysis
    print("\n📊 INDICATOR GROUP CORRELATIONS")
    print("-" * 50)
    
    # Define groups
    sectors = [c for c in corr_matrix.columns if c.startswith('xl') or c in ['iwm_us', 'gld_us']]
    rates = [c for c in corr_matrix.columns if c.startswith('DGS') or c in ['DFF', 'T10Y2Y', 'T10Y3M']]
    commodities = [c for c in corr_matrix.columns if 'OIL' in c or 'HHNGSP' in c or c == 'gld_us']
    
    if len(sectors) > 1:
        sector_corr = corr_matrix.loc[sectors, sectors]
        avg_sector_corr = sector_corr.values[np.triu_indices(len(sectors), k=1)].mean()
        print(f"  Avg sector correlation: {avg_sector_corr:.3f}")
    
    if len(rates) > 1:
        rates_corr = corr_matrix.loc[rates, rates]
        avg_rates_corr = rates_corr.values[np.triu_indices(len(rates), k=1)].mean()
        print(f"  Avg rates correlation: {avg_rates_corr:.3f}")
    
    # Save correlation matrix
    output_dir = OUTPUT_DIR / "correlation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
    print(f"\n💾 Saved full matrix to {output_dir / 'correlation_matrix.csv'}")
    
    # Create heatmap
    print("\n🎨 Generating heatmap...")
    
    # Use a subset for readability
    if len(corr_matrix) > 25:
        # Select most interesting indicators
        key_indicators = []
        # Add sectors
        key_indicators.extend([c for c in sectors if c in corr_matrix.columns][:8])
        # Add rates
        key_indicators.extend([c for c in ['DGS2', 'DGS10', 'DGS30', 'T10Y2Y', 'DFF'] if c in corr_matrix.columns])
        # Add VIX
        if 'VIXCLS' in corr_matrix.columns:
            key_indicators.append('VIXCLS')
        # Add spreads
        key_indicators.extend([c for c in ['BAMLH0A0HYM2', 'BAMLC0A4CBBB'] if c in corr_matrix.columns])
        # Add commodities
        key_indicators.extend([c for c in ['DCOILWTICO', 'gld_us'] if c in corr_matrix.columns])
        # Add FX
        key_indicators.extend([c for c in ['DTWEXBGS', 'DEXJPUS'] if c in corr_matrix.columns])
        
        key_indicators = list(dict.fromkeys(key_indicators))  # Remove duplicates, preserve order
        subset_corr = corr_matrix.loc[key_indicators, key_indicators]
    else:
        subset_corr = corr_matrix
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(subset_corr, dtype=bool), k=1)
    sns.heatmap(
        subset_corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={'size': 8}
    )
    
    ax.set_title('PRISM Correlation Matrix (10-Year Returns)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"💾 Saved heatmap to {output_dir / 'correlation_heatmap.png'}")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return corr_matrix


if __name__ == "__main__":
    main()