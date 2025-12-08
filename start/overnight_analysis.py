#!/usr/bin/env python3
"""
================================================================================
PRISM ENGINE: ULTIMATE OVERNIGHT ANALYSIS
================================================================================

This is the FULL computational beast - designed to run 6-12 hours overnight.
It combines everything: Monte Carlo simulation, Bootstrap confidence intervals,
Multi-resolution analysis, Cross-validation, and Statistical significance testing.

WHAT THIS DOES:

1. MULTI-RESOLUTION ANALYSIS
   - Runs all 16 lenses at 5 different time scales (21, 42, 63, 126, 252 days)
   - Shows which lenses work best at which frequency
   - Daily stepping for maximum granularity

2. BOOTSTRAP CONFIDENCE INTERVALS  
   - For each time window, resamples data 100x
   - Gives you confidence intervals on every lens score
   - Know which signals are statistically robust

3. MONTE CARLO NULL DISTRIBUTION
   - Generates 500 synthetic "random" market histories
   - Runs PRISM on each to build null distribution
   - Your real detections get p-values (statistical significance)

4. CROSS-INDICATOR VALIDATION
   - Tests all pairs of indicator groups
   - Finds which combinations best predict regime shifts
   - Builds a "lens importance" ranking with confidence

5. REGIME TRANSITION MATRIX
   - Hidden Markov Model trained on full history
   - Transition probabilities between regimes
   - Current regime probability with uncertainty

ESTIMATED RUNTIME:
- Multi-resolution (5 scales × daily): ~2-3 hours
- Bootstrap (100 resamples per window): ~3-4 hours  
- Monte Carlo (500 simulations): ~2-3 hours
- Total: 6-12 hours depending on hardware

OUTPUT FILES:
- prism_multiresolution.csv: All lens scores at all time scales
- prism_bootstrap_ci.csv: Confidence intervals for each detection
- prism_montecarlo_null.csv: Null distribution for significance testing
- prism_regime_transitions.csv: Markov transition matrix
- prism_significant_events.csv: Events with p-values < 0.05
- [Multiple PNG visualizations]

USAGE:
    python run_overnight_analysis.py

    # Or with nohup to run in background:
    nohup python run_overnight_analysis.py > prism_overnight.log 2>&1 &

Author: PRISM Engine
Date: December 2025
================================================================================
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import time
import warnings
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.utils import resample
from scipy import stats
from scipy.ndimage import gaussian_filter1d

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


# Optional HMM
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️  hmmlearn not installed - HMM analysis will be skipped")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use centralized DB path resolution
DB_PATH = get_db_path()

# ============================================================================
# CONFIGURATION - THE BEAST SETTINGS
# ============================================================================

CONFIG = {
    # Time range
    'years_back': 20,  # 2005-2025 (full coverage period)

    # Multi-resolution settings
    'window_sizes': [21, 42, 63, 126, 252],  # 1mo, 2mo, 3mo, 6mo, 1yr
    'step_size': 1,  # DAILY resolution (this is what makes it heavy)

    # Bootstrap settings
    'n_bootstrap': 100,  # Resamples per window for confidence intervals
    'bootstrap_confidence': 0.95,  # 95% CI

    # Monte Carlo settings
    'n_monte_carlo': 500,  # Synthetic histories to generate
    'mc_block_size': 21,  # Block bootstrap size for synthetic data

    # Regime detection
    'n_regimes': 3,
    'consensus_thresholds': [0.25, 0.30, 0.35, 0.40, 0.45],  # Test multiple (lowered)

    # Statistical significance
    'p_value_threshold': 0.05,

    # Performance
    'n_jobs': max(1, mp.cpu_count() - 1),  # Use all but one CPU core
    'chunk_size': 100,  # Process in chunks to manage memory
    'use_parallel': True,  # Enable parallel processing

    # What to run (set False to skip sections)
    'run_multiresolution': True,
    'run_bootstrap': True,
    'run_montecarlo': True,
    'run_hmm_analysis': True,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and prepare data from unified indicator_values table."""
    print("\n" + "=" * 70)
    print("📥 LOADING DATA")
    print("=" * 70)

    from data.sql.db_connector import load_all_indicators_wide

    # Use unified loader (falls back to legacy tables if needed)
    panel = load_all_indicators_wide()

    if panel.empty:
        print("   ⚠️ No data found in database")
        return pd.DataFrame(), pd.DataFrame()

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
# LENS IMPLEMENTATIONS (Optimized for speed)
# ============================================================================

def compute_all_lenses(returns_window, prices_window, n_regimes=3):
    """Compute all lens scores for a single window. Optimized version."""
    
    scores = {}
    n_samples, n_features = returns_window.shape
    
    # Skip if insufficient data
    if n_samples < 20 or n_features < 3:
        return {k: 0.0 for k in ['PCA', 'Correlation', 'Volatility', 'Skewness',
                                  'Kurtosis', 'Entropy', 'Anomaly', 'Clustering',
                                  'Regime_GMM', 'HMM', 'Magnitude', 'Drawdown',
                                  'Momentum', 'Network', 'Dispersion', 'Covariance']}
    
    returns_arr = returns_window.fillna(0).values
    
    # 1. PCA - variance concentration
    try:
        scaled = StandardScaler().fit_transform(returns_arr)
        pca = PCA(n_components=min(3, n_features, n_samples))
        pca.fit(scaled)
        scores['PCA'] = pca.explained_variance_ratio_[0]
    except:
        scores['PCA'] = 0.0
    
    # 2. Correlation - market coherence
    try:
        corr = np.corrcoef(returns_arr.T)
        upper = corr[np.triu_indices(len(corr), k=1)]
        valid = upper[~np.isnan(upper)]
        scores['Correlation'] = np.mean(np.abs(valid)) if len(valid) > 0 else 0.0
    except:
        scores['Correlation'] = 0.0
    
    # 3. Volatility
    scores['Volatility'] = np.nanstd(returns_arr, axis=0).mean()
    
    # 4. Skewness
    try:
        skews = stats.skew(returns_arr, axis=0, nan_policy='omit')
        scores['Skewness'] = -np.nanmean(skews)  # Flip so higher = more crash risk
    except:
        scores['Skewness'] = 0.0
    
    # 5. Kurtosis
    try:
        kurts = stats.kurtosis(returns_arr, axis=0, nan_policy='omit')
        scores['Kurtosis'] = max(0, np.nanmean(kurts))
    except:
        scores['Kurtosis'] = 0.0
    
    # 6. Entropy
    try:
        entropies = []
        for j in range(n_features):
            col = returns_arr[:, j]
            col = col[~np.isnan(col)]
            if len(col) > 20:
                hist, _ = np.histogram(col, bins=20, density=True)
                hist = hist[hist > 0]
                entropies.append(-np.sum(hist * np.log(hist + 1e-10)))
        scores['Entropy'] = np.mean(entropies) if entropies else 0.0
    except:
        scores['Entropy'] = 0.0
    
    # 7. Anomaly - Isolation Forest
    try:
        if n_samples < 30:
            scores['Anomaly'] = 0.0
        else:
            iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=1)
            scaled = StandardScaler().fit_transform(returns_arr)
            iso.fit(scaled)
            # Use decision_function for continuous anomaly scores
            # More negative = more anomalous
            decision_scores = iso.decision_function(scaled)
            # Return mean anomaly (flip sign so higher = more anomalous)
            scores['Anomaly'] = -np.mean(decision_scores)
    except:
        scores['Anomaly'] = 0.0
    
    # 8. Clustering
    try:
        kmeans = KMeans(n_clusters=min(n_regimes, n_samples // 5), 
                       random_state=42, n_init=5)
        kmeans.fit(StandardScaler().fit_transform(returns_arr))
        scores['Clustering'] = 1.0 / (1.0 + kmeans.inertia_ / n_samples)
    except:
        scores['Clustering'] = 0.0
    
    # 9. Regime GMM
    try:
        mean_ret = returns_arr.mean(axis=1).reshape(-1, 1)
        gmm = GaussianMixture(n_components=min(n_regimes, n_samples // 10),
                             random_state=42, n_init=2, max_iter=50)
        gmm.fit(mean_ret)
        probs = gmm.predict_proba(mean_ret)
        stress_idx = np.argmin(gmm.means_.flatten())
        scores['Regime_GMM'] = probs[:, stress_idx].mean()
    except:
        scores['Regime_GMM'] = 0.0
    
    # 10. HMM
    if HMM_AVAILABLE:
        try:
            mean_ret = returns_arr.mean(axis=1).reshape(-1, 1)
            hmm = GaussianHMM(n_components=2, covariance_type='diag',
                            n_iter=50, random_state=42)
            hmm.fit(mean_ret)
            states = hmm.predict(mean_ret)
            transitions = np.sum(states[1:] != states[:-1])
            scores['HMM'] = transitions / len(states)
        except:
            scores['HMM'] = 0.0
    else:
        scores['HMM'] = 0.0
    
    # 11. Magnitude
    scores['Magnitude'] = np.abs(returns_arr).mean()
    
    # 12. Drawdown
    try:
        # Use ffill properly (method= is deprecated)
        prices_filled = prices_window.ffill().bfill()
        if prices_filled.empty or prices_filled.isna().all().all():
            scores['Drawdown'] = 0.0
        else:
            prices_arr = prices_filled.values
            drawdowns = []
            for j in range(prices_arr.shape[1]):
                col = prices_arr[:, j]
                if np.isnan(col).all() or len(col) < 2:
                    continue
                # Replace any remaining NaN with forward fill
                col = pd.Series(col).ffill().bfill().values
                if col[0] > 0:  # Valid prices
                    peak = np.maximum.accumulate(col)
                    dd = (col[-1] - peak[-1]) / (peak[-1] + 1e-10)
                    drawdowns.append(dd)
            scores['Drawdown'] = -np.nanmean(drawdowns) if drawdowns else 0.0
    except Exception as e:
        scores['Drawdown'] = 0.0
    
    # 13. Momentum (autocorrelation)
    try:
        autocorrs = []
        for j in range(n_features):
            col = returns_arr[:, j]
            if len(col) > 10:
                ac = np.corrcoef(col[:-5], col[5:])[0, 1]
                if not np.isnan(ac):
                    autocorrs.append(abs(ac))
        scores['Momentum'] = np.mean(autocorrs) if autocorrs else 0.0
    except:
        scores['Momentum'] = 0.0
    
    # 14. Network density
    try:
        corr = np.corrcoef(returns_arr.T)
        significant = (np.abs(corr) > 0.5).sum() - n_features
        max_edges = n_features * (n_features - 1)
        scores['Network'] = significant / max_edges if max_edges > 0 else 0.0
    except:
        scores['Network'] = 0.0
    
    # 15. Dispersion
    scores['Dispersion'] = np.nanstd(returns_arr, axis=1).mean()
    
    # 16. Covariance regime
    try:
        mid = n_samples // 2
        if mid > 10:
            lw1 = LedoitWolf().fit(returns_arr[:mid])
            lw2 = LedoitWolf().fit(returns_arr[mid:])
            diff = np.linalg.norm(lw1.covariance_ - lw2.covariance_, 'fro')
            avg = (np.linalg.norm(lw1.covariance_, 'fro') + 
                   np.linalg.norm(lw2.covariance_, 'fro')) / 2
            scores['Covariance'] = diff / avg if avg > 0 else 0.0
        else:
            scores['Covariance'] = 0.0
    except:
        scores['Covariance'] = 0.0
    
    return scores


# ============================================================================
# MULTI-RESOLUTION ANALYSIS
# ============================================================================

def _process_window_full(args):
    """Process a single window for multiresolution analysis (picklable)."""
    ret_values, ret_columns, ret_index, price_values, price_columns, window_date, window_size, n_regimes = args

    # Reconstruct DataFrames
    ret_window = pd.DataFrame(ret_values, columns=ret_columns, index=ret_index)
    price_window = pd.DataFrame(price_values, columns=price_columns, index=ret_index)

    scores = compute_all_lenses(ret_window, price_window, n_regimes)
    scores['date'] = window_date
    scores['window_size'] = window_size

    return scores


def run_multiresolution_analysis(panel, returns, use_parallel=None):
    """Run all lenses at multiple time scales."""

    print("\n" + "=" * 70)
    print("🔬 MULTI-RESOLUTION ANALYSIS")
    print("=" * 70)
    print(f"   Window sizes: {CONFIG['window_sizes']}")
    print(f"   Step size: {CONFIG['step_size']} (daily)")

    if use_parallel is None:
        use_parallel = CONFIG.get('use_parallel', True)

    n_jobs = CONFIG['n_jobs']

    all_results = []

    for window_size in CONFIG['window_sizes']:
        print(f"\n   Processing window size: {window_size} days...")

        # Prepare all window args
        window_args = []
        for idx in range(window_size, len(returns), CONFIG['step_size']):
            ret_window = returns.iloc[idx-window_size:idx]
            price_window = panel.iloc[idx-window_size:idx]
            window_date = returns.index[idx]

            window_args.append((
                ret_window.values,
                ret_window.columns.tolist(),
                ret_window.index.tolist(),
                price_window.values,
                price_window.columns.tolist(),
                window_date,
                window_size,
                CONFIG['n_regimes']
            ))

        n_windows = len(window_args)
        print(f"      Windows to process: {n_windows}")

        start_time = time.time()

        if use_parallel and n_windows > 100:
            print(f"      Using {n_jobs} parallel workers...")
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                window_results = list(executor.map(_process_window_full, window_args, chunksize=50))
        else:
            # Sequential processing with progress
            window_results = []
            for i, args in enumerate(window_args):
                result = _process_window_full(args)
                window_results.append(result)

                if (i + 1) % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (n_windows - i - 1) / rate / 60
                    print(f"      {i+1}/{n_windows} ({100*(i+1)/n_windows:.1f}%) "
                          f"- {remaining:.1f} min remaining")

        all_results.extend(window_results)
        elapsed = time.time() - start_time
        print(f"      ✅ Completed in {elapsed/60:.1f} min")

        # Memory cleanup
        gc.collect()

    results_df = pd.DataFrame(all_results)
    return results_df


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def _bootstrap_single_window(args):
    """Bootstrap a single window - for parallel processing (picklable)."""
    ret_values, ret_columns, ret_index, price_values, price_columns, window_date, n_bootstrap, n_regimes = args

    # Reconstruct DataFrames
    returns_window = pd.DataFrame(ret_values, columns=ret_columns, index=ret_index)
    prices_window = pd.DataFrame(price_values, columns=price_columns, index=ret_index)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = resample(range(len(returns_window)), replace=True)
        ret_resampled = returns_window.iloc[idx]
        price_resampled = prices_window.iloc[idx]

        scores = compute_all_lenses(ret_resampled, price_resampled, n_regimes)
        bootstrap_scores.append(scores)

    # Compute confidence intervals
    bs_df = pd.DataFrame(bootstrap_scores)
    ci_low = bs_df.quantile(0.025)
    ci_high = bs_df.quantile(0.975)
    ci_mean = bs_df.mean()

    return window_date, ci_low.to_dict(), ci_mean.to_dict(), ci_high.to_dict()


def run_bootstrap_analysis(panel, returns, base_window=63, use_parallel=None):
    """Run bootstrap analysis for confidence intervals."""

    print("\n" + "=" * 70)
    print("📊 BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)
    print(f"   Bootstrap samples: {CONFIG['n_bootstrap']}")
    print(f"   Window size: {base_window}")

    if use_parallel is None:
        use_parallel = CONFIG.get('use_parallel', True)

    n_jobs = CONFIG['n_jobs']

    # Sample every 5 days for bootstrap (too expensive for daily)
    step = 5

    # Prepare all window args
    window_args = []
    for idx in range(base_window, len(returns), step):
        ret_window = returns.iloc[idx-base_window:idx]
        price_window = panel.iloc[idx-base_window:idx]
        window_date = returns.index[idx]

        window_args.append((
            ret_window.values,
            ret_window.columns.tolist(),
            ret_window.index.tolist(),
            price_window.values,
            price_window.columns.tolist(),
            window_date,
            CONFIG['n_bootstrap'],
            CONFIG['n_regimes']
        ))

    n_windows = len(window_args)
    print(f"   Windows to analyze: {n_windows}")

    start_time = time.time()

    if use_parallel and n_windows > 20:
        print(f"   Using {n_jobs} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            raw_results = list(executor.map(_bootstrap_single_window, window_args, chunksize=10))
    else:
        # Sequential processing with progress
        raw_results = []
        for i, args in enumerate(window_args):
            result = _bootstrap_single_window(args)
            raw_results.append(result)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (n_windows - i - 1) / rate / 60
                print(f"   {i+1}/{n_windows} ({100*(i+1)/n_windows:.1f}%) "
                      f"- {remaining:.1f} min remaining")

    # Convert raw results to DataFrame
    results = []
    for window_date, ci_low, ci_mean, ci_high in raw_results:
        result = {
            'date': window_date,
            **{f'{k}_mean': v for k, v in ci_mean.items()},
            **{f'{k}_ci_low': v for k, v in ci_low.items()},
            **{f'{k}_ci_high': v for k, v in ci_high.items()},
        }
        results.append(result)

    elapsed = time.time() - start_time
    print(f"   ✅ Completed in {elapsed/60:.1f} min")

    return pd.DataFrame(results)


# ============================================================================
# MONTE CARLO NULL DISTRIBUTION
# ============================================================================

def _generate_synthetic_returns(ret_values, ret_columns, n_samples, block_size, seed):
    """Generate synthetic returns using block bootstrap (picklable)."""
    np.random.seed(seed)

    n_obs = ret_values.shape[0]
    n_blocks = n_samples // block_size + 1

    synthetic = []
    for _ in range(n_blocks):
        # Random starting point
        start = np.random.randint(0, n_obs - block_size)
        block = ret_values[start:start+block_size]
        synthetic.append(block)

    synthetic = np.vstack(synthetic)[:n_samples]
    return pd.DataFrame(synthetic, columns=ret_columns)


def _run_single_mc_simulation(args):
    """Run a single Monte Carlo simulation (picklable)."""
    ret_values, ret_columns, ret_index, n_samples, base_window, block_size, n_regimes, sim_seed = args

    np.random.seed(sim_seed)

    # Generate synthetic data
    syn_returns = _generate_synthetic_returns(ret_values, ret_columns, n_samples, block_size, sim_seed)

    # Create synthetic prices (cumulative returns)
    syn_prices = (1 + syn_returns).cumprod() * 100
    syn_prices.index = ret_index[:len(syn_prices)]
    syn_returns.index = ret_index[:len(syn_returns)]

    # Run lenses on a sample of windows
    sample_windows = np.random.choice(
        range(base_window, len(syn_returns)),
        size=min(50, len(syn_returns) - base_window),
        replace=False
    )

    sim_scores = []
    for idx in sample_windows:
        ret_window = syn_returns.iloc[idx-base_window:idx]
        price_window = syn_prices.iloc[idx-base_window:idx]
        scores = compute_all_lenses(ret_window, price_window, n_regimes)
        sim_scores.append(scores)

    # Compute average consensus for this simulation
    sim_df = pd.DataFrame(sim_scores)

    # Normalize
    for col in sim_df.columns:
        min_val, max_val = sim_df[col].min(), sim_df[col].max()
        if max_val > min_val:
            sim_df[col] = (sim_df[col] - min_val) / (max_val - min_val)

    # Consensus = fraction above 0.5
    consensus = (sim_df > 0.5).sum(axis=1) / len(sim_df.columns)
    return consensus.values.tolist()


def run_montecarlo_analysis(panel, returns, base_window=63, use_parallel=None):
    """Generate null distribution via Monte Carlo simulation."""

    print("\n" + "=" * 70)
    print("🎲 MONTE CARLO NULL DISTRIBUTION")
    print("=" * 70)
    print(f"   Simulations: {CONFIG['n_monte_carlo']}")
    print(f"   Block size: {CONFIG['mc_block_size']}")

    if use_parallel is None:
        use_parallel = CONFIG.get('use_parallel', True)

    n_jobs = CONFIG['n_jobs']

    # Prepare simulation args
    sim_args = []
    for sim in range(CONFIG['n_monte_carlo']):
        sim_args.append((
            returns.values,
            returns.columns.tolist(),
            returns.index.tolist(),
            len(returns),
            base_window,
            CONFIG['mc_block_size'],
            CONFIG['n_regimes'],
            sim  # Use simulation number as seed for reproducibility
        ))

    start_time = time.time()

    if use_parallel and CONFIG['n_monte_carlo'] > 20:
        print(f"   Using {n_jobs} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            all_consensus = list(executor.map(_run_single_mc_simulation, sim_args, chunksize=10))
    else:
        # Sequential processing with progress
        all_consensus = []
        for sim, args in enumerate(sim_args):
            result = _run_single_mc_simulation(args)
            all_consensus.append(result)

            if (sim + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (sim + 1) / elapsed
                remaining = (CONFIG['n_monte_carlo'] - sim - 1) / rate / 60
                print(f"   Simulation {sim+1}/{CONFIG['n_monte_carlo']} "
                      f"({100*(sim+1)/CONFIG['n_monte_carlo']:.1f}%) "
                      f"- {remaining:.1f} min remaining")

    # Flatten all consensus values
    null_consensus = []
    for consensus_list in all_consensus:
        null_consensus.extend(consensus_list)

    elapsed = time.time() - start_time
    print(f"   ✅ Completed in {elapsed/60:.1f} min")

    null_consensus = np.array(null_consensus)

    # Compute percentiles for p-value calculation
    percentiles = {
        'p01': np.percentile(null_consensus, 99),
        'p05': np.percentile(null_consensus, 95),
        'p10': np.percentile(null_consensus, 90),
        'p25': np.percentile(null_consensus, 75),
        'p50': np.percentile(null_consensus, 50),
        'mean': null_consensus.mean(),
        'std': null_consensus.std(),
    }

    print(f"\n   Null distribution statistics:")
    print(f"      Mean consensus: {percentiles['mean']:.3f}")
    print(f"      Std consensus: {percentiles['std']:.3f}")
    print(f"      95th percentile: {percentiles['p05']:.3f}")
    print(f"      99th percentile: {percentiles['p01']:.3f}")

    return null_consensus, percentiles


# ============================================================================
# HMM REGIME ANALYSIS
# ============================================================================

def run_hmm_regime_analysis(returns):
    """Full HMM analysis with transition matrix."""
    
    if not HMM_AVAILABLE:
        print("\n⚠️  Skipping HMM analysis (hmmlearn not installed)")
        return None
    
    print("\n" + "=" * 70)
    print("🔄 HIDDEN MARKOV MODEL REGIME ANALYSIS")
    print("=" * 70)
    
    # Use mean return as observable
    obs = returns.mean(axis=1).values.reshape(-1, 1)
    
    # Fit HMM
    print("   Fitting HMM with 3 regimes...")
    hmm = GaussianHMM(n_components=3, covariance_type='full', 
                     n_iter=200, random_state=42)
    hmm.fit(obs)
    
    # Get state sequence
    states = hmm.predict(obs)
    state_probs = hmm.predict_proba(obs)
    
    # Transition matrix
    trans_mat = hmm.transmat_
    
    # State statistics
    state_means = hmm.means_.flatten()
    state_order = np.argsort(state_means)  # Order by mean return
    
    regime_names = ['Crisis', 'Normal', 'Bull']
    
    print(f"\n   Regime Statistics:")
    for i, idx in enumerate(state_order):
        pct_time = (states == idx).mean() * 100
        mean_ret = state_means[idx] * 252 * 100  # Annualized %
        print(f"      {regime_names[i]}: {pct_time:.1f}% of time, "
              f"mean return: {mean_ret:.1f}% annualized")
    
    print(f"\n   Transition Matrix (rows=from, cols=to):")
    print(f"      {'':10} {regime_names[0]:>10} {regime_names[1]:>10} {regime_names[2]:>10}")
    for i, idx in enumerate(state_order):
        row = trans_mat[idx, state_order]
        print(f"      {regime_names[i]:10} {row[0]:10.3f} {row[1]:10.3f} {row[2]:10.3f}")
    
    # Current regime
    current_state = states[-1]
    current_probs = state_probs[-1, state_order]
    print(f"\n   Current Regime Probabilities:")
    for i, name in enumerate(regime_names):
        print(f"      {name}: {current_probs[i]:.1%}")
    
    results = {
        'states': pd.Series(states, index=returns.index),
        'state_probs': pd.DataFrame(state_probs[:, state_order], 
                                    index=returns.index,
                                    columns=regime_names),
        'transition_matrix': pd.DataFrame(trans_mat[state_order][:, state_order],
                                         index=regime_names, columns=regime_names),
        'state_means': pd.Series(state_means[state_order], index=regime_names),
    }
    
    return results


# ============================================================================
# STATISTICAL SIGNIFICANCE
# ============================================================================

def compute_significance(real_consensus, null_distribution):
    """Compute p-values for detected events."""
    
    print("\n" + "=" * 70)
    print("📈 STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 70)
    
    # For each real consensus value, compute p-value
    # p-value = fraction of null values >= observed value
    p_values = []
    for val in real_consensus:
        p = (null_distribution >= val).mean()
        p_values.append(p)
    
    p_values = np.array(p_values)
    
    n_significant_05 = (p_values < 0.05).sum()
    n_significant_01 = (p_values < 0.01).sum()
    
    print(f"   Total observations: {len(p_values)}")
    print(f"   Significant at p<0.05: {n_significant_05} ({100*n_significant_05/len(p_values):.1f}%)")
    print(f"   Significant at p<0.01: {n_significant_01} ({100*n_significant_01/len(p_values):.1f}%)")
    
    return p_values


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_visualizations(results, output_dir):
    """Create all visualization outputs."""
    
    print("\n" + "=" * 70)
    print("🎨 GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Unpack results
    multiresolution = results.get('multiresolution')
    bootstrap = results.get('bootstrap')
    montecarlo = results.get('montecarlo')
    hmm_results = results.get('hmm')
    p_values = results.get('p_values')
    consensus_by_window = results.get('consensus_by_window', {})
    
    # =========================================================================
    # FIGURE 1: Multi-Resolution Heatmap
    # =========================================================================
    if multiresolution is not None:
        print("   Creating multi-resolution heatmap...")
        
        fig, axes = plt.subplots(len(CONFIG['window_sizes']), 1, 
                                figsize=(20, 4 * len(CONFIG['window_sizes'])),
                                sharex=True)
        
        for i, window_size in enumerate(CONFIG['window_sizes']):
            ax = axes[i]
            
            # Filter to this window size
            df = multiresolution[multiresolution['window_size'] == window_size].copy()
            df = df.set_index('date')
            
            # Get lens columns
            lens_cols = [c for c in df.columns if c != 'window_size']
            
            # Normalize
            for col in lens_cols:
                min_val, max_val = df[col].min(), df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            
            # Resample to weekly for visualization
            weekly = df[lens_cols].resample('W').mean()
            
            # Plot
            im = ax.imshow(weekly.T, aspect='auto', cmap='RdYlBu_r',
                          vmin=0, vmax=1, interpolation='nearest')
            
            ax.set_yticks(range(len(lens_cols)))
            ax.set_yticklabels(lens_cols, fontsize=8)
            ax.set_title(f'Window Size: {window_size} days', fontsize=11)
            
            # X-axis
            n_weeks = len(weekly)
            tick_positions = np.linspace(0, n_weeks-1, 10).astype(int)
            tick_labels = [weekly.index[i].strftime('%Y-%m') for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
        
        axes[-1].set_xlabel('Date', fontsize=11)
        fig.suptitle('PRISM Multi-Resolution Analysis: Lens Activity Across Time Scales',
                    fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=axes, shrink=0.6, label='Normalized Score')
        plt.tight_layout()
        fig.savefig(output_dir / 'multiresolution_heatmap.png', dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: multiresolution_heatmap.png")
        plt.close()
    
    # =========================================================================
    # FIGURE 2: Consensus with Confidence Intervals
    # =========================================================================
    if bootstrap is not None and len(consensus_by_window) > 0:
        print("   Creating consensus with confidence intervals...")
        
        fig, ax = plt.subplots(figsize=(18, 8))
        
        # Use 63-day window consensus
        if 63 in consensus_by_window:
            consensus = consensus_by_window[63]
            dates = consensus.index
            
            ax.fill_between(dates, consensus, alpha=0.4, color='darkred')
            ax.plot(dates, consensus, 'darkred', linewidth=1, label='Consensus (63d window)')
            
            # Mark significant events if we have p-values
            if p_values is not None and len(p_values) == len(consensus):
                sig_mask = p_values < 0.05
                ax.scatter(dates[sig_mask], consensus.values[sig_mask], 
                          color='red', s=20, zorder=5, label='p < 0.05')
            
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
            ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
            
            ax.set_ylabel('Consensus Score', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title('PRISM Consensus Signal with Statistical Significance\n'
                        'Red dots = Significant regime shifts (p < 0.05)',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(output_dir / 'consensus_significance.png', dpi=150, bbox_inches='tight')
            print(f"   💾 Saved: consensus_significance.png")
            plt.close()
    
    # =========================================================================
    # FIGURE 3: Monte Carlo Null Distribution
    # =========================================================================
    if montecarlo is not None:
        print("   Creating Monte Carlo null distribution...")
        
        null_dist, percentiles = montecarlo
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(null_dist, bins=50, density=True, alpha=0.7, color='steelblue',
               label='Null Distribution')
        
        ax.axvline(x=percentiles['p05'], color='orange', linestyle='--',
                  label=f'95th percentile: {percentiles["p05"]:.3f}')
        ax.axvline(x=percentiles['p01'], color='red', linestyle='--',
                  label=f'99th percentile: {percentiles["p01"]:.3f}')
        
        ax.set_xlabel('Consensus Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Monte Carlo Null Distribution\n'
                    'What consensus looks like by random chance',
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(output_dir / 'montecarlo_null.png', dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: montecarlo_null.png")
        plt.close()
    
    # =========================================================================
    # FIGURE 4: HMM Regime Analysis
    # =========================================================================
    if hmm_results is not None:
        print("   Creating HMM regime analysis...")
        
        fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        
        # State probabilities over time
        ax1 = axes[0]
        probs = hmm_results['state_probs']
        ax1.stackplot(probs.index, probs.T, labels=probs.columns,
                     colors=['#d62728', '#2ca02c', '#1f77b4'], alpha=0.8)
        ax1.set_ylabel('Regime Probability', fontsize=11)
        ax1.set_title('Hidden Markov Model: Regime Probabilities Over Time',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, 1)
        
        # State sequence
        ax2 = axes[1]
        states = hmm_results['states']
        ax2.fill_between(states.index, states, alpha=0.7, step='post')
        ax2.set_ylabel('Regime State', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Crisis', 'Normal', 'Bull'])
        
        plt.tight_layout()
        fig.savefig(output_dir / 'hmm_regimes.png', dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: hmm_regimes.png")
        plt.close()
    
    print("   ✅ Visualizations complete")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PRISM Engine: Ultimate Overnight Analysis',
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
    parser.add_argument(
        '--skip-multiresolution', action='store_true',
        help='Skip multi-resolution analysis'
    )
    parser.add_argument(
        '--skip-bootstrap', action='store_true',
        help='Skip bootstrap analysis'
    )
    parser.add_argument(
        '--skip-montecarlo', action='store_true',
        help='Skip Monte Carlo analysis'
    )
    parser.add_argument(
        '--skip-hmm', action='store_true',
        help='Skip HMM regime analysis'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Update config from CLI args
    CONFIG['use_parallel'] = args.parallel
    if args.jobs is not None:
        CONFIG['n_jobs'] = args.jobs
    if args.skip_multiresolution:
        CONFIG['run_multiresolution'] = False
    if args.skip_bootstrap:
        CONFIG['run_bootstrap'] = False
    if args.skip_montecarlo:
        CONFIG['run_montecarlo'] = False
    if args.skip_hmm:
        CONFIG['run_hmm_analysis'] = False

    print("=" * 80)
    print("=" * 80)
    print("   PRISM ENGINE: ULTIMATE OVERNIGHT ANALYSIS")
    print("=" * 80)
    print("=" * 80)

    start_time = time.time()
    print(f"\n🚀 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n📋 Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")

    # Load data
    panel, returns = load_data()

    # Results container
    results = {}

    use_parallel = CONFIG['use_parallel']

    # =========================================================================
    # 1. MULTI-RESOLUTION ANALYSIS
    # =========================================================================
    if CONFIG['run_multiresolution']:
        results['multiresolution'] = run_multiresolution_analysis(panel, returns, use_parallel=use_parallel)

        # Compute consensus for each window size
        consensus_by_window = {}
        for window_size in CONFIG['window_sizes']:
            df = results['multiresolution']
            df_window = df[df['window_size'] == window_size].copy()
            df_window = df_window.set_index('date')

            lens_cols = [c for c in df_window.columns if c != 'window_size']

            # Normalize
            for col in lens_cols:
                min_val, max_val = df_window[col].min(), df_window[col].max()
                if max_val > min_val:
                    df_window[col] = (df_window[col] - min_val) / (max_val - min_val)

            # Consensus
            consensus = (df_window[lens_cols] > 0.5).sum(axis=1) / len(lens_cols)
            consensus_by_window[window_size] = consensus

        results['consensus_by_window'] = consensus_by_window

    # =========================================================================
    # 2. BOOTSTRAP ANALYSIS
    # =========================================================================
    if CONFIG['run_bootstrap']:
        results['bootstrap'] = run_bootstrap_analysis(panel, returns, base_window=63, use_parallel=use_parallel)

    # =========================================================================
    # 3. MONTE CARLO ANALYSIS
    # =========================================================================
    if CONFIG['run_montecarlo']:
        null_dist, percentiles = run_montecarlo_analysis(panel, returns, base_window=63, use_parallel=use_parallel)
        results['montecarlo'] = (null_dist, percentiles)

        # Compute p-values for real data
        if 63 in results.get('consensus_by_window', {}):
            real_consensus = results['consensus_by_window'][63].values
            results['p_values'] = compute_significance(real_consensus, null_dist)

    # =========================================================================
    # 4. HMM REGIME ANALYSIS
    # =========================================================================
    if CONFIG['run_hmm_analysis']:
        results['hmm'] = run_hmm_regime_analysis(returns)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_dir = OUTPUT_DIR / "overnight_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("💾 SAVING RESULTS")
    print("=" * 70)
    
    if results.get('multiresolution') is not None:
        results['multiresolution'].to_csv(output_dir / 'prism_multiresolution.csv', index=False)
        print(f"   Saved: prism_multiresolution.csv")
    
    if results.get('bootstrap') is not None:
        results['bootstrap'].to_csv(output_dir / 'prism_bootstrap_ci.csv', index=False)
        print(f"   Saved: prism_bootstrap_ci.csv")
    
    if results.get('montecarlo') is not None:
        null_dist, percentiles = results['montecarlo']
        pd.DataFrame({'null_consensus': null_dist}).to_csv(
            output_dir / 'prism_montecarlo_null.csv', index=False
        )
        pd.DataFrame([percentiles]).to_csv(
            output_dir / 'prism_montecarlo_percentiles.csv', index=False
        )
        print(f"   Saved: prism_montecarlo_null.csv, prism_montecarlo_percentiles.csv")
    
    if results.get('hmm') is not None:
        results['hmm']['states'].to_csv(output_dir / 'prism_hmm_states.csv')
        results['hmm']['state_probs'].to_csv(output_dir / 'prism_hmm_probs.csv')
        results['hmm']['transition_matrix'].to_csv(output_dir / 'prism_hmm_transitions.csv')
        print(f"   Saved: prism_hmm_*.csv")
    
    if results.get('p_values') is not None and 63 in results.get('consensus_by_window', {}):
        sig_df = pd.DataFrame({
            'date': results['consensus_by_window'][63].index,
            'consensus': results['consensus_by_window'][63].values,
            'p_value': results['p_values']
        })
        sig_df.to_csv(output_dir / 'prism_significance.csv', index=False)
        
        # Save significant events only
        sig_events = sig_df[sig_df['p_value'] < 0.05].sort_values('consensus', ascending=False)
        sig_events.to_csv(output_dir / 'prism_significant_events.csv', index=False)
        print(f"   Saved: prism_significance.csv, prism_significant_events.csv")
        print(f"   Significant events (p<0.05): {len(sig_events)}")
    
    # =========================================================================
    # CREATE VISUALIZATIONS
    # =========================================================================
    create_comprehensive_visualizations(results, output_dir)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print("   ANALYSIS COMPLETE")
    print("=" * 80)
    print("=" * 80)
    
    print(f"\n⏱️  Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"📁 Output directory: {output_dir}")
    
    if results.get('p_values') is not None:
        n_sig = (results['p_values'] < 0.05).sum()
        n_total = len(results['p_values'])
        print(f"\n🎯 Statistically Significant Regime Shifts Detected: {n_sig}")
        print(f"   ({100*n_sig/n_total:.1f}% of observations)")
    
    if results.get('hmm') is not None:
        probs = results['hmm']['state_probs'].iloc[-1]
        print(f"\n🔮 Current Market Regime (HMM):")
        for regime, prob in probs.items():
            print(f"   {regime}: {prob:.1%}")
    
    print(f"\n✅ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
