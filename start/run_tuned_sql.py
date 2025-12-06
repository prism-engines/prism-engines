#!/usr/bin/env python3
"""
PRISM Tuned Analysis (SQL-Integrated)
======================================

Reads calibration from SQL, runs analysis, saves results back to SQL.

This is the production version - all scripts should follow this pattern.

Usage:
    python run_tuned_sql.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    import os
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from output_config import OUTPUT_DIR, DATA_DIR

# Import the SQL interface
sys.path.insert(0, str(Path(__file__).parent))
from sql_schema_extension import PrismDB

OUTPUT_PATH = OUTPUT_DIR / "tuned_analysis"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


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
    return pd.Series(np.abs(pca.components_[0]), index=clean.columns)

def lens_momentum(returns: pd.DataFrame) -> pd.Series:
    return ((1 + returns).prod() - 1).abs()

def lens_beta(returns: pd.DataFrame) -> pd.Series:
    market_col = None
    for col in returns.columns:
        if 'sp500' in col.lower():
            market_col = col
            break
    if not market_col:
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
        s = returns[col].dropna()
        if len(s) < 50:
            continue
        try:
            std = s.std()
            if std == 0:
                continue
            cumdev = (s - s.mean()).cumsum()
            result[col] = (cumdev.max() - cumdev.min()) / std / np.sqrt(len(s))
        except:
            pass
    return pd.Series(result)

def lens_correlation(returns: pd.DataFrame) -> pd.Series:
    return returns.corr().abs().mean()

def lens_dispersion(returns: pd.DataFrame) -> pd.Series:
    cross_std = returns.std(axis=1)
    return pd.Series({col: abs(returns[col].corr(cross_std)) 
                      for col in returns.columns 
                      if not np.isnan(returns[col].corr(cross_std))})

def lens_skewness(returns: pd.DataFrame) -> pd.Series:
    return -returns.skew()

def lens_autocorrelation(returns: pd.DataFrame) -> pd.Series:
    return pd.Series({col: abs(returns[col].autocorr(lag=1)) 
                      for col in returns.columns 
                      if not np.isnan(returns[col].autocorr(lag=1))})

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
# MAIN
# =============================================================================

def run_tuned_sql():
    """Run tuned analysis using SQL for config and results."""
    
    print("=" * 70)
    print("🏎️  PRISM TUNED ANALYSIS (SQL-Integrated)")
    print("=" * 70)
    print()
    
    # Initialize database connection
    db = PrismDB()
    
    # Load calibration from SQL
    print("📥 Loading calibration from SQL...")
    calibration = db.get_calibration()
    
    lens_weights = calibration['lens_weights']
    active_indicators = calibration['active_indicators']
    thresholds = calibration['thresholds'] or {'rank_warning': 10, 'rank_danger': 5}
    windows_config = calibration['windows'] or {'multi_window': [21, 63, 126]}
    
    if not lens_weights:
        print("❌ No calibration in SQL. Run:")
        print("   python start/sql_schema_extension.py")
        return
    
    print(f"   ✅ Loaded {len(lens_weights)} lenses, {len(active_indicators)} indicators")
    print(f"   ✅ Version: {calibration['version']}")
    
    # Load data
    print("\n📊 Loading market data...")
    db_path = DATA_DIR / "prism.db"
    if not db_path.exists():
        db_path = Path.home() / "prism_data" / "prism.db"
    
    conn = sqlite3.connect(db_path)
    
    market = pd.read_sql("SELECT date, ticker, value FROM market_prices WHERE value IS NOT NULL", conn)
    econ = pd.read_sql("SELECT date, series_id, value FROM econ_values WHERE value IS NOT NULL", conn)
    conn.close()
    
    market_wide = market.pivot(index='date', columns='ticker', values='value')
    econ_wide = econ.pivot(index='date', columns='series_id', values='value')
    panel = pd.concat([market_wide, econ_wide], axis=1)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    
    # Filter to active indicators
    available = [i for i in active_indicators if i in panel.columns]
    panel = panel[available]
    
    print(f"   ✅ {len(available)} active indicators loaded")
    print(f"   ✅ Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    
    # Run multi-window analysis
    windows = windows_config.get('multi_window', [21, 63, 126])
    print(f"\n🔄 Running multi-window analysis: {windows}")
    
    all_results = []
    
    for window in windows:
        returns = panel.pct_change().tail(window * 2).dropna(how='all')
        recent = returns.tail(window)
        
        lens_results = {}
        for lens_name, weight in lens_weights.items():
            if lens_name in LENS_FUNCTIONS:
                try:
                    scores = LENS_FUNCTIONS[lens_name](recent)
                    if len(scores) > 0:
                        lens_results[lens_name] = scores
                except:
                    pass
        
        if lens_results:
            combined = pd.DataFrame(lens_results)
            ranked = combined.rank(ascending=False)
            
            weighted_rank = pd.Series(0.0, index=ranked.index)
            total_weight = 0
            
            for lens in ranked.columns:
                w = lens_weights.get(lens, 1.0)
                weighted_rank += ranked[lens].fillna(ranked[lens].max()) * w
                total_weight += w
            
            weighted_rank = weighted_rank / total_weight
            
            for indicator in weighted_rank.index:
                all_results.append({
                    'indicator': indicator,
                    'window': window,
                    'weighted_rank': weighted_rank[indicator],
                })
    
    results_df = pd.DataFrame(all_results)
    
    # Aggregate
    agg = results_df.groupby('indicator').agg({
        'weighted_rank': 'mean',
    }).sort_values('weighted_rank')
    
    agg['consensus_rank'] = agg['weighted_rank'].rank()
    
    # Add indicator metadata
    for indicator in agg.index:
        config = db.get_indicator_config(indicator)
        if config:
            agg.loc[indicator, 'tier'] = config.get('tier')
            agg.loc[indicator, 'optimal_window'] = config.get('optimal_window')
    
    # Generate signals
    warning_map = {
        'T10Y2Y': ('Yield Curve 10Y-2Y', 'Recession indicator'),
        'T10Y3M': ('Yield Curve 10Y-3M', 'Recession indicator'),
        'BAMLH0A0HYM2': ('HY Credit', 'Credit stress'),
        'BAMLC0A4CBBB': ('BBB Credit', 'Corporate stress'),
        'VIXCLS': ('VIX', 'Fear index'),
        'DGS10': ('10Y Treasury', 'Rate stress'),
        'gld_us': ('Gold', 'Safe haven'),
    }
    
    signals = []
    for ind, (name, meaning) in warning_map.items():
        if ind in agg.index:
            rank = agg.loc[ind, 'consensus_rank']
            if rank <= thresholds['rank_danger']:
                status = 'DANGER'
            elif rank <= thresholds['rank_warning']:
                status = 'WARNING'
            else:
                status = 'NORMAL'
            signals.append({
                'indicator': ind,
                'name': name,
                'meaning': meaning,
                'rank': rank,
                'status': status,
            })
    
    signals_df = pd.DataFrame(signals).sort_values('rank')
    
    # Save results to SQL
    print("\n💾 Saving results to SQL...")
    db.save_rankings(agg, analysis_type='tuned')
    db.save_signals(signals_df)
    print("   ✅ Rankings saved")
    print("   ✅ Signals saved")
    
    # Also save to CSV for visibility
    agg.to_csv(OUTPUT_PATH / "rankings.csv")
    signals_df.to_csv(OUTPUT_PATH / "signals.csv")
    
    # Print results
    print("\n" + "=" * 70)
    print("🎯 TOP 15 INDICATORS")
    print("=" * 70)
    
    for i, (ind, row) in enumerate(agg.head(15).iterrows(), 1):
        tier = int(row.get('tier', 0)) if pd.notna(row.get('tier')) else '?'
        print(f"   {i:2}. {ind:25} | Rank: {row['consensus_rank']:4.0f} | T{tier}")
    
    # Signal summary
    print("\n" + "=" * 70)
    print("🚦 SIGNALS")
    print("=" * 70)
    
    danger = signals_df[signals_df['status'] == 'DANGER']
    warning = signals_df[signals_df['status'] == 'WARNING']
    
    if len(danger) > 0:
        print("\n🔴 DANGER:")
        for _, row in danger.iterrows():
            print(f"   • {row['name']} - Rank {row['rank']:.0f}")
    
    if len(warning) > 0:
        print("\n🟡 WARNING:")
        for _, row in warning.iterrows():
            print(f"   • {row['name']} - Rank {row['rank']:.0f}")
    
    if len(danger) == 0 and len(warning) == 0:
        print("\n✅ All clear - no warning signals")
    
    # Show history if available
    print("\n📈 Recent DANGER signal history:")
    history = db.get_danger_history(days=30)
    if not history.empty:
        for _, row in history.head(5).iterrows():
            print(f"   {row['run_date']}: {row['indicator']} - Rank {row['rank']:.0f}")
    else:
        print("   No prior DANGER signals in last 30 days")
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE - Results saved to SQL")
    print("=" * 70)
    
    return agg, signals_df


if __name__ == "__main__":
    run_tuned_sql()
