#!/usr/bin/env python3
"""
PRISM Fully Tuned Analysis
==========================

The complete calibrated PRISM engine that uses:
- Optimized lens weights (from Part 1)
- Curated indicators (from Part 2)  
- Optimal windows per indicator (from Part 3)

This is the production-ready analysis.

Usage:
    python run_tuned.py
    
Output:
    tuned_analysis/
        rankings.csv
        signals.csv
        summary.png
        report.md
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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from output_config import OUTPUT_DIR, DATA_DIR
from data.sql.db_path import get_db_path

CALIBRATION_DIR = OUTPUT_DIR / "calibration"
OUTPUT_PATH = OUTPUT_DIR / "tuned_analysis"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOAD CALIBRATION
# =============================================================================

def load_master_config() -> dict:
    """Load the master calibration config."""
    config_path = CALIBRATION_DIR / "master_config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        print("⚠️ No master config found - run calibration_unified.py first")
        return None


# =============================================================================
# LENS IMPLEMENTATIONS
# =============================================================================

def lens_pca(returns: pd.DataFrame) -> pd.Series:
    from sklearn.decomposition import PCA
    clean = returns.dropna(axis=1, thresh=int(len(returns) * 0.8)).fillna(0)
    if clean.shape[1] < 3:
        return pd.Series(dtype=float)
    pca = PCA(n_components=min(3, clean.shape[1]))
    pca.fit(clean)
    loadings = np.abs(pca.components_[0])
    return pd.Series(loadings, index=clean.columns)


def lens_momentum(returns: pd.DataFrame) -> pd.Series:
    cumret = (1 + returns).prod() - 1
    return cumret.abs()


def lens_beta(returns: pd.DataFrame) -> pd.Series:
    market_col = None
    for col in returns.columns:
        if 'sp500' in col.lower() or 'spy' in col.lower():
            market_col = col
            break
    if market_col is None:
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
    return pd.Series(betas)


def lens_persistence(returns: pd.DataFrame) -> pd.Series:
    result = {}
    for col in returns.columns:
        series = returns[col].dropna()
        if len(series) < 50:
            continue
        try:
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            cumdev = (series - mean).cumsum()
            R = cumdev.max() - cumdev.min()
            result[col] = R / std / np.sqrt(len(series))
        except:
            pass
    return pd.Series(result)


def lens_correlation(returns: pd.DataFrame) -> pd.Series:
    corr = returns.corr().abs()
    return corr.mean()


def lens_dispersion(returns: pd.DataFrame) -> pd.Series:
    cross_std = returns.std(axis=1)
    result = {}
    for col in returns.columns:
        try:
            c = returns[col].corr(cross_std)
            if not np.isnan(c):
                result[col] = abs(c)
        except:
            pass
    return pd.Series(result)


def lens_skewness(returns: pd.DataFrame) -> pd.Series:
    return -returns.skew()


def lens_autocorrelation(returns: pd.DataFrame) -> pd.Series:
    result = {}
    for col in returns.columns:
        try:
            ac = returns[col].autocorr(lag=1)
            if not np.isnan(ac):
                result[col] = abs(ac)
        except:
            pass
    return pd.Series(result)


LENS_FUNCTIONS = {
    'pca': lens_pca,
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
    """Load panel using central runtime loader."""
    from panel.runtime_loader import load_calibrated_panel

    try:
        panel = load_calibrated_panel()
        if not panel.empty:
            return panel
    except Exception as e:
        import warnings
        warnings.warn(f"Central loader failed: {e}", RuntimeWarning)

    # Fallback to direct db_connector call
    from data.sql.db_connector import load_all_indicators_wide
    panel = load_all_indicators_wide()

    if panel.empty:
        import warnings
        warnings.warn(
            "indicator_values table is empty. Data may need migration.",
            DeprecationWarning
        )

    return panel


# =============================================================================
# TUNED ANALYSIS
# =============================================================================

def run_tuned_analysis():
    """Run the fully tuned PRISM analysis."""
    
    print("=" * 70)
    print("🏎️  PRISM FULLY TUNED ANALYSIS")
    print("=" * 70)
    print()
    
    # Load calibration
    config = load_master_config()
    if config is None:
        print("❌ Cannot run without calibration. Run these first:")
        print("   python start/calibration_lens_benchmark.py")
        print("   python start/calibration_indicator_curation.py")
        print("   python start/calibration_window_optimization.py")
        print("   python start/calibration_unified.py")
        return
    
    print(f"📅 Config version: {config['version']}")
    print(f"   Generated: {config['generated']}")
    
    # Get active lenses and weights
    active_lenses = {lens: info['weight'] 
                     for lens, info in config['lenses'].items() 
                     if info['use']}
    
    # Get active indicators
    active_indicators = [ind for ind, info in config['indicators'].items() 
                        if info['use']]
    
    print(f"\n🔬 Active lenses: {len(active_lenses)}")
    for lens, weight in sorted(active_lenses.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {lens}: {weight:.2f}")
    
    print(f"\n📊 Active indicators: {len(active_indicators)}")
    
    # Load data
    print("\n📥 Loading data...")
    panel = load_panel()
    
    # Filter to active indicators
    available = [ind for ind in active_indicators if ind in panel.columns]
    panel_filtered = panel[available]
    
    print(f"   Using {len(available)} of {len(active_indicators)} active indicators")
    print(f"   Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    
    # Run multi-window analysis
    windows = config['windows']['multi_window']
    print(f"\n🔄 Running multi-window analysis: {windows}")
    
    all_results = []
    
    for window in windows:
        print(f"\n   Window: {window} days")
        
        # Get recent returns for this window
        returns = panel_filtered.pct_change().tail(window * 2).dropna(how='all')
        recent_returns = returns.tail(window)
        
        # Run each lens
        lens_results = {}
        
        for lens_name, weight in active_lenses.items():
            if lens_name not in LENS_FUNCTIONS:
                continue
            
            func = LENS_FUNCTIONS[lens_name]
            
            try:
                scores = func(recent_returns)
                if len(scores) > 0:
                    lens_results[lens_name] = scores
            except Exception as e:
                pass
        
        # Combine with weights
        if lens_results:
            combined = pd.DataFrame(lens_results)
            
            # Rank within each lens
            ranked = combined.rank(ascending=False)
            
            # Weight the ranks
            weighted_rank = pd.Series(0.0, index=ranked.index)
            total_weight = 0
            
            for lens_name in ranked.columns:
                w = active_lenses.get(lens_name, 1.0)
                weighted_rank += ranked[lens_name].fillna(ranked[lens_name].max()) * w
                total_weight += w
            
            weighted_rank = weighted_rank / total_weight
            
            for indicator in weighted_rank.index:
                all_results.append({
                    'indicator': indicator,
                    'window': window,
                    'weighted_rank': weighted_rank[indicator],
                    'final_rank': weighted_rank.rank()[indicator],
                })
    
    results_df = pd.DataFrame(all_results)
    
    # Aggregate across windows
    print("\n⚖️  Aggregating across windows...")
    
    agg = results_df.groupby('indicator').agg({
        'weighted_rank': 'mean',
        'final_rank': 'mean',
    }).sort_values('weighted_rank')
    
    agg['consensus_rank'] = agg['weighted_rank'].rank()
    
    # Get indicator metadata
    for indicator in agg.index:
        if indicator in config['indicators']:
            info = config['indicators'][indicator]
            agg.loc[indicator, 'tier'] = info['tier']
            agg.loc[indicator, 'optimal_window'] = info['optimal_window']
            agg.loc[indicator, 'type'] = info['type']
    
    # Print results
    print("\n" + "=" * 70)
    print("🎯 TOP 20 INDICATORS (Fully Calibrated)")
    print("=" * 70)
    
    top_20 = agg.head(20)
    
    for i, (indicator, row) in enumerate(top_20.iterrows(), 1):
        tier = int(row.get('tier', 0))
        tier_str = f"T{tier}" if tier > 0 else "?"
        window = int(row.get('optimal_window', 63))
        
        print(f"   {i:2}. {indicator:25} | Rank: {row['consensus_rank']:4.0f} | "
              f"{tier_str} | {window}d")
    
    # Save results
    print("\n💾 Saving results...")
    
    agg.to_csv(OUTPUT_PATH / "rankings.csv")
    print(f"   ✅ {OUTPUT_PATH / 'rankings.csv'}")
    
    # Generate signals
    signals = generate_signals(agg, config)
    signals.to_csv(OUTPUT_PATH / "signals.csv")
    print(f"   ✅ {OUTPUT_PATH / 'signals.csv'}")
    
    # Create visualization
    create_tuned_visualization(agg, config, panel_filtered)
    
    # Signal summary
    print("\n" + "=" * 70)
    print("🚦 SIGNAL STATUS")
    print("=" * 70)
    
    warning_threshold = config['thresholds']['rank_warning']
    danger_threshold = config['thresholds']['rank_danger']
    
    danger_signals = signals[signals['status'] == 'DANGER']
    warning_signals = signals[signals['status'] == 'WARNING']
    
    if len(danger_signals) > 0:
        print("\n🔴 DANGER SIGNALS:")
        for _, row in danger_signals.iterrows():
            print(f"   • {row['indicator']} ({row['name']}) - Rank {row['rank']:.0f}")
    
    if len(warning_signals) > 0:
        print("\n🟡 WARNING SIGNALS:")
        for _, row in warning_signals.iterrows():
            print(f"   • {row['indicator']} ({row['name']}) - Rank {row['rank']:.0f}")
    
    if len(danger_signals) == 0 and len(warning_signals) == 0:
        print("\n✅ No warning signals - system shows normal/low risk")
    
    print("\n" + "=" * 70)
    print("✅ TUNED ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📁 Results: {OUTPUT_PATH}")
    
    return agg, signals


def generate_signals(rankings: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate trading signals from rankings."""
    
    warning_map = {
        'T10Y2Y': ('Yield Curve 10Y-2Y', 'Recession indicator'),
        'T10Y3M': ('Yield Curve 10Y-3M', 'Recession indicator'),
        'BAMLH0A0HYM2': ('HY Credit Spread', 'Credit stress'),
        'BAMLC0A4CBBB': ('BBB Credit Spread', 'Corporate stress'),
        'VIXCLS': ('VIX', 'Fear index'),
        'DGS10': ('10Y Treasury', 'Rate stress'),
        'DGS2': ('2Y Treasury', 'Short rate stress'),
        'TEDRATE': ('TED Spread', 'Banking stress'),
        'DCOILWTICO': ('Oil (WTI)', 'Energy/inflation'),
        'gld_us': ('Gold', 'Safe haven demand'),
    }
    
    warning_threshold = config['thresholds']['rank_warning']
    danger_threshold = config['thresholds']['rank_danger']
    
    signals = []
    
    for indicator, (name, meaning) in warning_map.items():
        if indicator in rankings.index:
            rank = rankings.loc[indicator, 'consensus_rank']
            
            if rank <= danger_threshold:
                status = 'DANGER'
            elif rank <= warning_threshold:
                status = 'WARNING'
            else:
                status = 'NORMAL'
            
            signals.append({
                'indicator': indicator,
                'name': name,
                'meaning': meaning,
                'rank': rank,
                'status': status,
            })
    
    return pd.DataFrame(signals).sort_values('rank')


def create_tuned_visualization(rankings: pd.DataFrame, config: dict, panel: pd.DataFrame):
    """Create visualization of tuned analysis."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Top 20 indicators
    ax1 = axes[0, 0]
    
    top_20 = rankings.head(20)
    
    colors = []
    for _, row in top_20.iterrows():
        tier = row.get('tier', 3)
        if tier == 1:
            colors.append('#27AE60')
        elif tier == 2:
            colors.append('#F39C12')
        else:
            colors.append('#E74C3C')
    
    y_pos = range(len(top_20))
    ax1.barh(y_pos, top_20['consensus_rank'], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_20.index)
    ax1.invert_yaxis()
    ax1.invert_xaxis()  # Lower rank = better = longer bar
    ax1.set_xlabel('Consensus Rank (lower = more important)')
    ax1.set_title('Top 20 Indicators (Fully Calibrated)', fontweight='bold')
    
    # 2. Lens weights
    ax2 = axes[0, 1]
    
    lens_weights = {l: info['weight'] for l, info in config['lenses'].items() if info['use']}
    sorted_lenses = sorted(lens_weights.items(), key=lambda x: x[1], reverse=True)
    
    names = [x[0] for x in sorted_lenses]
    weights = [x[1] for x in sorted_lenses]
    
    ax2.barh(names, weights, color='steelblue', alpha=0.7)
    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Weight')
    ax2.set_title('Calibrated Lens Weights', fontweight='bold')
    
    # 3. Warning indicator status
    ax3 = axes[1, 0]
    
    warning_indicators = ['T10Y2Y', 'T10Y3M', 'BAMLH0A0HYM2', 'VIXCLS', 'gld_us']
    warning_names = ['Yield Curve\n(10Y-2Y)', 'Yield Curve\n(10Y-3M)', 
                     'HY Credit', 'VIX', 'Gold']
    
    ranks = []
    colors3 = []
    
    for ind in warning_indicators:
        if ind in rankings.index:
            r = rankings.loc[ind, 'consensus_rank']
            ranks.append(r)
            if r <= 5:
                colors3.append('#E74C3C')
            elif r <= 10:
                colors3.append('#F39C12')
            else:
                colors3.append('#27AE60')
        else:
            ranks.append(30)
            colors3.append('#95A5A6')
    
    ax3.barh(warning_names, [30 - r for r in ranks], color=colors3)
    ax3.set_xlabel('Warning Level (higher = more concern)')
    ax3.set_xlim(0, 30)
    ax3.set_title('Warning Indicator Status', fontweight='bold')
    
    # Threshold lines
    ax3.axvline(25, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(20, color='orange', linestyle='--', alpha=0.5)
    
    # 4. Top indicator chart
    ax4 = axes[1, 1]
    
    top_indicator = rankings.index[0]
    if top_indicator in panel.columns:
        recent = panel[top_indicator].tail(252)
        ax4.plot(recent.index, recent.values, color='steelblue', linewidth=2)
        ax4.set_title(f'#1 Indicator: {top_indicator}', fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('PRISM Fully Tuned Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PATH / "summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ {OUTPUT_PATH / 'summary.png'}")


if __name__ == "__main__":
    run_tuned_analysis()
