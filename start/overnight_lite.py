#!/usr/bin/env python3
"""
PRISM Overnight Analysis - LITE VERSION
========================================

Runs core analysis without the HMM-heavy Monte Carlo that caused 30+ hour runs.

Runtime: ~30-60 minutes instead of 30+ hours

What it does:
1. Multi-resolution analysis (5 window sizes)
2. Bootstrap confidence intervals (50 resamples)
3. Basic regime detection
4. Consensus ranking across time

What it skips:
- HMM fitting in Monte Carlo (the bottleneck)
- 500 synthetic simulations

Usage:
    python start/overnight_lite.py
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.utils import resample
from scipy import stats

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

# Use centralized DB path resolution
DB_PATH = get_db_path()

# ============================================================================
# CONFIGURATION - LITE VERSION
# ============================================================================

CONFIG = {
    'years_back': 10,  # 10 years instead of 20
    'window_sizes': [21, 63, 126, 252],  # Skip 42-day
    'step_size': 5,  # Weekly steps instead of daily
    'n_bootstrap': 50,  # 50 instead of 100
    'bootstrap_confidence': 0.95,
    'n_regimes': 3,
    'n_jobs': max(1, mp.cpu_count() - 1),
    'use_parallel': True,  # Enable parallel processing
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and prepare data using unified indicator_values table."""
    print("\n" + "=" * 60)
    print("📥 LOADING DATA")
    print("=" * 60)

    from data.sql.db_connector import load_all_indicators_wide

    panel = load_all_indicators_wide()

    if panel.empty:
        import warnings
        warnings.warn(
            "indicator_values table is empty. Data may need migration.",
            DeprecationWarning
        )
        return pd.DataFrame(), pd.DataFrame()

    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # Filter to analysis period
    cutoff = panel.index.max() - pd.DateOffset(years=CONFIG['years_back'])
    panel = panel[panel.index >= cutoff]

    # Keep columns with 50%+ coverage
    valid_cols = panel.columns[panel.notna().sum() > len(panel) * 0.5]
    panel = panel[valid_cols].ffill().bfill().dropna(axis=1)

    # Compute returns
    returns = panel.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"   Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"   Indicators: {len(panel.columns)}")
    print(f"   Trading days: {len(returns)}")

    return panel, returns


# ============================================================================
# CORE LENS FUNCTIONS (Simplified)
# ============================================================================

def compute_pca_importance(returns):
    """PCA-based importance."""
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(returns)
        pca = PCA(n_components=min(5, len(returns.columns)))
        pca.fit(scaled)
        loadings = np.abs(pca.components_).mean(axis=0)
        return pd.Series(loadings, index=returns.columns)
    except:
        return pd.Series(1.0, index=returns.columns)


def compute_volatility_importance(returns):
    """Volatility-based importance."""
    return returns.std()


def compute_correlation_importance(returns):
    """Correlation centrality."""
    corr = returns.corr().abs()
    return corr.mean()


def compute_anomaly_importance(returns):
    """Isolation Forest anomaly scores."""
    try:
        iso = IsolationForest(n_estimators=50, random_state=42)
        scores = []
        for col in returns.columns:
            iso.fit(returns[[col]])
            scores.append(-iso.score_samples(returns[[col]]).mean())
        return pd.Series(scores, index=returns.columns)
    except:
        return pd.Series(1.0, index=returns.columns)


def compute_all_lenses(returns_window):
    """Compute all lens scores for a window."""
    results = {}
    
    results['pca'] = compute_pca_importance(returns_window)
    results['volatility'] = compute_volatility_importance(returns_window)
    results['correlation'] = compute_correlation_importance(returns_window)
    results['anomaly'] = compute_anomaly_importance(returns_window)
    
    # Combine into rankings
    rankings = pd.DataFrame(results)
    
    # Rank each lens (higher score = higher rank)
    for col in rankings.columns:
        rankings[f'{col}_rank'] = rankings[col].rank(ascending=False)
    
    # Consensus = mean rank
    rank_cols = [c for c in rankings.columns if '_rank' in c]
    rankings['consensus_rank'] = rankings[rank_cols].mean(axis=1)
    
    return rankings


# ============================================================================
# MULTI-RESOLUTION ANALYSIS
# ============================================================================

def _process_single_window(args):
    """Process a single window for multiresolution analysis (picklable for multiprocessing)."""
    window_returns_values, window_returns_columns, window_date, window_size = args

    # Reconstruct DataFrame from values (pickle-friendly)
    window_returns = pd.DataFrame(window_returns_values, columns=window_returns_columns)

    rankings = compute_all_lenses(window_returns)
    rankings['date'] = window_date
    rankings['window_size'] = window_size
    rankings['indicator'] = rankings.index

    return rankings


def run_multiresolution(panel, returns, use_parallel=None):
    """Run analysis at multiple time scales."""
    print("\n" + "=" * 60)
    print("🔬 MULTI-RESOLUTION ANALYSIS")
    print("=" * 60)

    if use_parallel is None:
        use_parallel = CONFIG.get('use_parallel', True)

    n_jobs = CONFIG['n_jobs']

    all_results = []

    for window_size in CONFIG['window_sizes']:
        print(f"\n   Window: {window_size} days")

        # Prepare all window args
        window_args = []
        for i in range(0, len(returns) - window_size, CONFIG['step_size']):
            window_returns = returns.iloc[i:i+window_size]
            window_date = returns.index[i + window_size]
            # Pass values and columns separately for pickle compatibility
            window_args.append((
                window_returns.values,
                window_returns.columns.tolist(),
                window_date,
                window_size
            ))

        n_windows = len(window_args)
        print(f"   Processing {n_windows} windows...")

        start_time = time.time()

        if use_parallel and n_windows > 10:
            print(f"   Using {n_jobs} parallel workers...")
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                window_results = list(executor.map(_process_single_window, window_args))
        else:
            # Sequential processing
            window_results = []
            for j, args in enumerate(window_args):
                result = _process_single_window(args)
                window_results.append(result)
                if (j + 1) % 100 == 0:
                    print(f"      {j+1}/{n_windows} windows processed...")

        if window_results:
            df = pd.concat(window_results, ignore_index=True)
            all_results.append(df)
            elapsed = time.time() - start_time
            print(f"   ✅ {len(window_results)} windows completed in {elapsed:.1f}s")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def _bootstrap_single(args):
    """Single bootstrap iteration (picklable for multiprocessing)."""
    returns_values, returns_columns, seed = args
    np.random.seed(seed)

    # Reconstruct DataFrame from values
    returns_window = pd.DataFrame(returns_values, columns=returns_columns)

    # Resample rows with replacement
    boot_returns = resample(returns_window, random_state=seed)
    rankings = compute_all_lenses(boot_returns)
    return rankings['consensus_rank'].to_dict()


def run_bootstrap(panel, returns, window_size=63, use_parallel=None):
    """Bootstrap confidence intervals for consensus rankings."""
    print("\n" + "=" * 60)
    print("📊 BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 60)

    if use_parallel is None:
        use_parallel = CONFIG.get('use_parallel', True)

    n_jobs = CONFIG['n_jobs']

    # Use most recent window
    recent_returns = returns.iloc[-window_size:]
    print(f"   Window: last {window_size} days")
    print(f"   Resamples: {CONFIG['n_bootstrap']}")

    # Prepare args with pickle-friendly data
    args = [
        (recent_returns.values, recent_returns.columns.tolist(), i)
        for i in range(CONFIG['n_bootstrap'])
    ]

    start_time = time.time()

    if use_parallel and CONFIG['n_bootstrap'] > 10:
        print(f"   Using {n_jobs} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            boot_results = list(executor.map(_bootstrap_single, args))
    else:
        boot_results = []
        for arg in args:
            result = _bootstrap_single(arg)
            boot_results.append(result)

    # Combine results
    boot_df = pd.DataFrame(boot_results)

    # Compute confidence intervals
    ci_low = (1 - CONFIG['bootstrap_confidence']) / 2
    ci_high = 1 - ci_low

    summary = pd.DataFrame({
        'indicator': boot_df.columns,
        'mean_rank': boot_df.mean(),
        'std_rank': boot_df.std(),
        'ci_lower': boot_df.quantile(ci_low),
        'ci_upper': boot_df.quantile(ci_high),
    })

    summary = summary.sort_values('mean_rank')
    elapsed = time.time() - start_time
    print(f"   ✅ Bootstrap complete in {elapsed:.1f}s")

    return summary


# ============================================================================
# REGIME DETECTION
# ============================================================================

def detect_regimes(returns):
    """Simple regime detection using clustering."""
    print("\n" + "=" * 60)
    print("🎯 REGIME DETECTION")
    print("=" * 60)
    
    # Rolling volatility as regime indicator
    vol = returns.std(axis=1).rolling(21).mean().dropna()
    
    # Cluster into regimes
    vol_2d = vol.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=CONFIG['n_regimes'], random_state=42, n_init=10)
    labels = kmeans.fit_predict(vol_2d)
    
    # Map to regime names based on volatility level
    centers = kmeans.cluster_centers_.flatten()
    regime_map = {i: rank for rank, i in enumerate(np.argsort(centers))}
    regime_names = {0: 'Low Vol', 1: 'Normal', 2: 'High Vol'}
    
    regimes = pd.Series(labels, index=vol.index)
    regimes = regimes.map(regime_map).map(regime_names)
    
    # Current regime
    current = regimes.iloc[-1]
    print(f"   Current regime: {current}")
    
    # Regime distribution
    dist = regimes.value_counts(normalize=True)
    for regime, pct in dist.items():
        print(f"   {regime}: {pct:.1%}")
    
    return regimes


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results, bootstrap, regimes, output_dir):
    """Create summary visualizations."""
    print("\n" + "=" * 60)
    print("📈 CREATING VISUALIZATIONS")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Top indicators by consensus
    fig, ax = plt.subplots(figsize=(12, 6))
    top_10 = bootstrap.head(10)
    ax.barh(top_10['indicator'], top_10['mean_rank'])
    ax.errorbar(top_10['mean_rank'], top_10['indicator'], 
                xerr=[top_10['mean_rank'] - top_10['ci_lower'],
                      top_10['ci_upper'] - top_10['mean_rank']],
                fmt='none', color='black', capsize=3)
    ax.set_xlabel('Mean Consensus Rank (lower = more important)')
    ax.set_title('Top 10 Indicators with 95% CI')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'top_indicators.png', dpi=150)
    plt.close()
    print("   ✅ top_indicators.png")
    
    # 2. Regime timeline
    fig, ax = plt.subplots(figsize=(14, 4))
    regime_colors = {'Low Vol': 'green', 'Normal': 'blue', 'High Vol': 'red'}
    for regime, color in regime_colors.items():
        mask = regimes == regime
        ax.scatter(regimes.index[mask], [1]*mask.sum(), 
                   c=color, label=regime, alpha=0.5, s=10)
    ax.legend()
    ax.set_title('Regime Timeline')
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_dir / 'regime_timeline.png', dpi=150)
    plt.close()
    print("   ✅ regime_timeline.png")
    
    # 3. Multi-resolution heatmap (if results exist)
    if not results.empty and 'window_size' in results.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, ws in enumerate(CONFIG['window_sizes'][:4]):
            ax = axes[idx // 2, idx % 2]
            ws_data = results[results['window_size'] == ws]
            
            if not ws_data.empty:
                # Get top 10 indicators for this window size
                top_inds = ws_data.groupby('indicator')['consensus_rank'].mean().nsmallest(10).index
                
                pivot = ws_data[ws_data['indicator'].isin(top_inds)].pivot(
                    index='date', columns='indicator', values='consensus_rank'
                )
                
                if not pivot.empty:
                    im = ax.imshow(pivot.T, aspect='auto', cmap='RdYlGn_r')
                    ax.set_title(f'{ws}-day window')
                    ax.set_ylabel('Indicator')
                    ax.set_yticks(range(len(pivot.columns)))
                    ax.set_yticklabels(pivot.columns, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'multiresolution_heatmap.png', dpi=150)
        plt.close()
        print("   ✅ multiresolution_heatmap.png")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PRISM Overnight Analysis - Lite Version',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--parallel', dest='parallel', action='store_true', default=True,
        help='Enable parallel processing (default: enabled)'
    )
    parser.add_argument(
        '--no-parallel', dest='parallel', action='store_false',
        help='Disable parallel processing for debugging'
    )
    parser.add_argument(
        '--jobs', '-j', type=int, default=None,
        help=f'Number of parallel workers (default: {mp.cpu_count() - 1})'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Update config from CLI args
    CONFIG['use_parallel'] = args.parallel
    if args.jobs is not None:
        CONFIG['n_jobs'] = args.jobs

    print("=" * 60)
    print("🚀 PRISM OVERNIGHT ANALYSIS - LITE")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Config: {CONFIG['years_back']}yr, step={CONFIG['step_size']}, bootstrap={CONFIG['n_bootstrap']}")
    print(f"Parallel: {CONFIG['use_parallel']} (workers: {CONFIG['n_jobs']})")

    start_time = time.time()

    # Load data
    panel, returns = load_data()

    if len(returns) < 100:
        print("❌ Not enough data!")
        return

    # Run analyses
    results = run_multiresolution(panel, returns, use_parallel=CONFIG['use_parallel'])
    bootstrap = run_bootstrap(panel, returns, window_size=63, use_parallel=CONFIG['use_parallel'])
    regimes = detect_regimes(returns)
    
    # Save results
    output_dir = OUTPUT_DIR / "overnight_lite"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results.empty:
        results.to_csv(output_dir / 'multiresolution.csv', index=False)
        print(f"\n   ✅ Saved: multiresolution.csv")
    
    bootstrap.to_csv(output_dir / 'bootstrap_ci.csv', index=False)
    print(f"   ✅ Saved: bootstrap_ci.csv")
    
    regimes.to_frame('regime').to_csv(output_dir / 'regimes.csv')
    print(f"   ✅ Saved: regimes.csv")
    
    # Visualizations
    create_visualizations(results, bootstrap, regimes, output_dir)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"   Runtime: {elapsed/60:.1f} minutes")
    print(f"   Output: {output_dir}")
    
    # Print top indicators
    print("\n📊 TOP 10 INDICATORS:")
    for i, row in bootstrap.head(10).iterrows():
        print(f"   {row['indicator']:20s} rank: {row['mean_rank']:.1f} ± {row['std_rank']:.1f}")


if __name__ == "__main__":
    main()
