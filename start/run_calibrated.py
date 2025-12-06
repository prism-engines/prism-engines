#!/usr/bin/env python3
"""
PRISM Calibrated Analysis
=========================

Runs PRISM with calibrated lens weights from the meta-engine.

This is the "tuned Ferrari" - same engine, optimized parameters.

Key differences from uncalibrated:
1. Lenses weighted by predictive performance
2. Weak lenses (volatility) downweighted
3. Strong lenses (momentum, beta, persistence) upweighted

Usage:
    python run_calibrated.py
    
Output:
    calibrated_analysis/rankings.csv - Weighted indicator rankings
    calibrated_analysis/signals.csv - Current warning signals
    calibrated_analysis/summary.png - Visualization
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    import os
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

import json
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from output_config import OUTPUT_DIR, DATA_DIR

# Import calibration
sys.path.insert(0, str(Path(__file__).parent))
from calibration_loader import (
    get_lens_weights, 
    get_top_lenses,
    apply_weights_to_rankings,
    get_consensus_events
)

OUTPUT_PATH = OUTPUT_DIR / "calibrated_analysis"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LENS IMPLEMENTATIONS (same as calibration, but we only run the good ones)
# =============================================================================

def lens_pca_variance(returns: pd.DataFrame) -> pd.Series:
    from sklearn.decomposition import PCA
    clean = returns.dropna(axis=1, thresh=int(len(returns) * 0.8)).fillna(0)
    if clean.shape[1] < 3:
        return pd.Series(dtype=float)
    pca = PCA(n_components=min(3, clean.shape[1]))
    pca.fit(clean)
    loadings = np.abs(pca.components_[0])
    return pd.Series(loadings, index=clean.columns).rank(ascending=False)


def lens_momentum(returns: pd.DataFrame) -> pd.Series:
    cumret = (1 + returns).prod() - 1
    return cumret.abs().rank(ascending=False)


def lens_beta(returns: pd.DataFrame, market_col: str = None) -> pd.Series:
    if market_col is None:
        for col in returns.columns:
            if 'sp500' in col.lower() or 'spy' in col.lower():
                market_col = col
                break
    if market_col is None or market_col not in returns.columns:
        return pd.Series(dtype=float)
    market = returns[market_col]
    betas = {}
    for col in returns.columns:
        if col == market_col:
            continue
        try:
            clean = pd.concat([returns[col], market], axis=1).dropna()
            if len(clean) > 20:
                slope, _, _, _, _ = stats.linregress(clean.iloc[:, 1], clean.iloc[:, 0])
                betas[col] = abs(slope)
        except:
            pass
    if not betas:
        return pd.Series(dtype=float)
    return pd.Series(betas).rank(ascending=False)


def lens_persistence(returns: pd.DataFrame) -> pd.Series:
    hurst_approx = {}
    for col in returns.columns:
        series = returns[col].dropna()
        if len(series) < 50:
            continue
        try:
            n = len(series)
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            cumdev = (series - mean).cumsum()
            R = cumdev.max() - cumdev.min()
            S = std
            if S > 0:
                hurst_approx[col] = R / S / np.sqrt(n)
        except:
            pass
    if not hurst_approx:
        return pd.Series(dtype=float)
    return pd.Series(hurst_approx).rank(ascending=False)


def lens_correlation(returns: pd.DataFrame) -> pd.Series:
    corr = returns.corr().abs()
    centrality = corr.mean()
    return centrality.rank(ascending=False)


def lens_dispersion(returns: pd.DataFrame) -> pd.Series:
    cross_std = returns.std(axis=1)
    corr_with_stress = {}
    for col in returns.columns:
        try:
            c = returns[col].corr(cross_std)
            if not np.isnan(c):
                corr_with_stress[col] = abs(c)
        except:
            pass
    if not corr_with_stress:
        return pd.Series(dtype=float)
    return pd.Series(corr_with_stress).rank(ascending=False)


def lens_skewness(returns: pd.DataFrame) -> pd.Series:
    skew = returns.skew()
    return (-skew).rank(ascending=False)


def lens_autocorrelation(returns: pd.DataFrame) -> pd.Series:
    autocorr = {}
    for col in returns.columns:
        try:
            ac = returns[col].autocorr(lag=1)
            if not np.isnan(ac):
                autocorr[col] = abs(ac)
        except:
            pass
    if not autocorr:
        return pd.Series(dtype=float)
    return pd.Series(autocorr).rank(ascending=False)


LENS_FUNCTIONS = {
    'pca': lens_pca_variance,
    'momentum': lens_momentum,
    'beta': lens_beta,
    'persistence': lens_persistence,
    'correlation': lens_correlation,
    'dispersion': lens_dispersion,
    'skewness': lens_skewness,
    'autocorrelation': lens_autocorrelation,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_panel() -> pd.DataFrame:
    db_path = DATA_DIR / "prism.db"
    if not db_path.exists():
        db_path = Path.home() / "prism_data" / "prism.db"
    
    conn = sqlite3.connect(db_path)
    
    market = pd.read_sql("""
        SELECT date, ticker, value 
        FROM market_prices 
        WHERE value IS NOT NULL
    """, conn)
    
    econ = pd.read_sql("""
        SELECT date, series_id, value 
        FROM econ_values 
        WHERE value IS NOT NULL
    """, conn)
    
    conn.close()
    
    market_wide = market.pivot(index='date', columns='ticker', values='value')
    econ_wide = econ.pivot(index='date', columns='series_id', values='value')
    
    panel = pd.concat([market_wide, econ_wide], axis=1)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    
    # Last 5 years for analysis
    cutoff = panel.index.max() - pd.Timedelta(days=5*365)
    panel = panel.loc[cutoff:]
    
    return panel


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_calibrated_analysis():
    """Run PRISM with calibrated weights."""
    
    print("=" * 70)
    print("🏎️  PRISM CALIBRATED ANALYSIS")
    print("=" * 70)
    print("   Using optimized lens weights from calibration")
    print()
    
    # Load calibration
    weights = get_lens_weights()
    top_lenses = get_top_lenses(6)  # Use top 6 lenses
    
    print("📊 Calibrated Lens Weights:")
    for lens in top_lenses:
        w = weights.get(lens, 1.0)
        bar = "█" * int(w * 5)
        print(f"   {lens:20} {w:.2f} {bar}")
    
    # Load data
    print("\n📥 Loading data...")
    panel = load_panel()
    print(f"   {panel.shape[0]} days × {panel.shape[1]} indicators")
    print(f"   Range: {panel.index.min().date()} to {panel.index.max().date()}")
    
    # Compute returns
    returns = panel.pct_change().dropna(how='all')
    
    # Use last 63 days for current analysis
    window = 63
    recent_returns = returns.tail(window)
    
    print(f"\n🔬 Running calibrated lenses on last {window} days...")
    
    # Run each lens
    lens_results = {}
    
    for lens_name in top_lenses:
        if lens_name in LENS_FUNCTIONS:
            func = LENS_FUNCTIONS[lens_name]
            try:
                rankings = func(recent_returns)
                if len(rankings) > 0:
                    lens_results[lens_name] = rankings
                    print(f"   ✅ {lens_name}: {len(rankings)} indicators ranked")
            except Exception as e:
                print(f"   ❌ {lens_name}: {e}")
    
    # Combine into DataFrame
    rankings_df = pd.DataFrame(lens_results)
    
    # Apply calibrated weights
    print("\n⚖️  Applying calibrated weights...")
    weighted_rankings = apply_weights_to_rankings(rankings_df)
    
    # Get top indicators
    print("\n🎯 TOP 15 INDICATORS (Weighted by Calibration):")
    print("-" * 50)
    
    top_15 = weighted_rankings.head(15)
    
    for i, (indicator, row) in enumerate(top_15.iterrows(), 1):
        final_rank = row['final_rank']
        weighted = row['weighted_rank']
        print(f"   {i:2}. {indicator:25} (weighted score: {weighted:.1f})")
    
    # Save rankings
    weighted_rankings.to_csv(OUTPUT_PATH / "rankings.csv")
    print(f"\n💾 Saved: {OUTPUT_PATH / 'rankings.csv'}")
    
    # Generate signals
    signals = generate_signals(weighted_rankings, weights)
    signals.to_csv(OUTPUT_PATH / "signals.csv")
    print(f"💾 Saved: {OUTPUT_PATH / 'signals.csv'}")
    
    # Create visualization
    create_summary_plot(weighted_rankings, weights, panel)
    
    # Print signal summary
    print("\n" + "=" * 70)
    print("🚦 CURRENT SIGNAL STATUS")
    print("=" * 70)
    
    # Check for warning indicators in top 10
    warning_indicators = ['T10Y2Y', 'BAMLH0A0HYM2', 'VIXCLS', 'DGS10', 'DGS2']
    top_10_indicators = set(weighted_rankings.head(10).index)
    
    warnings_active = []
    for ind in warning_indicators:
        if ind in top_10_indicators:
            rank = weighted_rankings.loc[ind, 'final_rank']
            warnings_active.append((ind, rank))
    
    if warnings_active:
        print("\n⚠️  WARNING INDICATORS IN TOP 10:")
        for ind, rank in warnings_active:
            print(f"   🔴 {ind} at rank {rank:.0f}")
        print("\n   Historical pattern: these lead market stress by 2-6 months")
    else:
        print("\n✅ No major warning indicators in Top 10")
        print("   System shows normal/low risk regime")
    
    print("\n" + "=" * 70)
    print("✅ CALIBRATED ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults: {OUTPUT_PATH}")
    
    return weighted_rankings, signals


def generate_signals(rankings: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Generate trading/warning signals from rankings."""
    
    signals = []
    
    # Warning indicators
    warning_map = {
        'T10Y2Y': ('Yield Curve', 'Bond market stress'),
        'BAMLH0A0HYM2': ('HY Credit', 'Credit market stress'),
        'BAMLC0A4CBBB': ('BBB Credit', 'Corporate credit stress'),
        'VIXCLS': ('VIX', 'Volatility spike'),
        'DGS10': ('10Y Yield', 'Rate stress'),
        'DGS2': ('2Y Yield', 'Short rate stress'),
    }
    
    for indicator, (name, meaning) in warning_map.items():
        if indicator in rankings.index:
            rank = rankings.loc[indicator, 'final_rank']
            weighted = rankings.loc[indicator, 'weighted_rank']
            
            if rank <= 5:
                status = 'DANGER'
            elif rank <= 10:
                status = 'WARNING'
            elif rank <= 15:
                status = 'ELEVATED'
            else:
                status = 'NORMAL'
            
            signals.append({
                'indicator': indicator,
                'name': name,
                'meaning': meaning,
                'rank': rank,
                'weighted_score': weighted,
                'status': status,
            })
    
    return pd.DataFrame(signals).sort_values('rank')


def create_summary_plot(rankings: pd.DataFrame, weights: dict, panel: pd.DataFrame):
    """Create summary visualization."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Top 15 indicators bar chart
    ax1 = axes[0, 0]
    top_15 = rankings.head(15)
    colors = ['#E74C3C' if i < 5 else '#F39C12' if i < 10 else '#27AE60' 
              for i in range(15)]
    
    y_pos = range(len(top_15))
    ax1.barh(y_pos, top_15['weighted_rank'], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_15.index)
    ax1.invert_yaxis()
    ax1.set_xlabel('Weighted Rank Score')
    ax1.set_title('Top 15 Indicators (Calibrated)', fontweight='bold')
    
    # Add rank numbers
    for i, (idx, row) in enumerate(top_15.iterrows()):
        ax1.text(row['weighted_rank'] + 0.5, i, f"#{int(row['final_rank'])}", 
                va='center', fontsize=9)
    
    # 2. Lens weights used
    ax2 = axes[0, 1]
    lens_names = list(weights.keys())
    lens_vals = list(weights.values())
    sorted_idx = np.argsort(lens_vals)[::-1]
    
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(lens_names)))
    ax2.barh([lens_names[i] for i in sorted_idx], 
             [lens_vals[i] for i in sorted_idx],
             color=[colors2[i] for i in sorted_idx])
    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Weight')
    ax2.set_title('Calibrated Lens Weights', fontweight='bold')
    
    # 3. Warning indicators status
    ax3 = axes[1, 0]
    warning_indicators = ['T10Y2Y', 'BAMLH0A0HYM2', 'BAMLC0A4CBBB', 'VIXCLS', 'DGS10']
    warning_names = ['Yield Curve', 'HY Credit', 'BBB Credit', 'VIX', '10Y Yield']
    
    ranks = []
    for ind in warning_indicators:
        if ind in rankings.index:
            ranks.append(rankings.loc[ind, 'final_rank'])
        else:
            ranks.append(30)
    
    colors3 = ['#E74C3C' if r <= 5 else '#F39C12' if r <= 10 else '#27AE60' 
               for r in ranks]
    
    ax3.barh(warning_names, [30 - r for r in ranks], color=colors3)  # Invert so higher = more warning
    ax3.set_xlabel('Warning Level (higher = more concern)')
    ax3.set_title('Warning Indicator Status', fontweight='bold')
    ax3.set_xlim(0, 30)
    
    # Add threshold lines
    ax3.axvline(25, color='red', linestyle='--', alpha=0.5, label='Danger')
    ax3.axvline(20, color='orange', linestyle='--', alpha=0.5, label='Warning')
    
    # 4. Recent price action of top indicator
    ax4 = axes[1, 1]
    top_indicator = rankings.index[0]
    
    if top_indicator in panel.columns:
        recent = panel[top_indicator].tail(252)
        ax4.plot(recent.index, recent.values, linewidth=2)
        ax4.set_title(f'Top Indicator: {top_indicator}', fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, f'No data for {top_indicator}', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Top Indicator', fontweight='bold')
    
    plt.suptitle('PRISM Calibrated Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PATH / "summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {OUTPUT_PATH / 'summary.png'}")


if __name__ == "__main__":
    run_calibrated_analysis()
