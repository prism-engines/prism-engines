#!/usr/bin/env python3
"""
PRISM Calibration Engine - Step 1: Lens Benchmark
==================================================

This is the first piece of the meta-engine that tunes PRISM.

What it does:
1. Runs each lens individually across historical data
2. Finds "consensus spikes" (when multiple lenses suddenly agree)
3. Scores each lens on how EARLY it detected these spikes
4. Outputs optimal lens weights

The key insight: we don't define what a "regime break" is.
We find moments when lenses agreed, then score who saw it first.

Usage:
    python calibration_lens_benchmark.py
    
Output:
    calibration/lens_scores.csv
    calibration/lens_weights.json
    calibration/consensus_events.csv
"""

import sys
from pathlib import Path

# Double-click support
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
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from output_config import OUTPUT_DIR, DATA_DIR

# Calibration output directory
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LENS IMPLEMENTATIONS (simplified for speed)
# =============================================================================

def lens_pca_variance(returns: pd.DataFrame) -> pd.Series:
    """PCA lens: rank by contribution to first principal component."""
    from sklearn.decomposition import PCA
    
    clean = returns.dropna(axis=1, thresh=int(len(returns) * 0.8))
    clean = clean.fillna(0)
    
    if clean.shape[1] < 3:
        return pd.Series(dtype=float)
    
    pca = PCA(n_components=min(3, clean.shape[1]))
    pca.fit(clean)
    
    # Importance = absolute loading on PC1
    loadings = np.abs(pca.components_[0])
    importance = pd.Series(loadings, index=clean.columns)
    
    return importance.rank(ascending=False)


def lens_volatility(returns: pd.DataFrame) -> pd.Series:
    """Volatility lens: rank by recent volatility spike."""
    vol = returns.std()
    return vol.rank(ascending=False)


def lens_correlation_centrality(returns: pd.DataFrame) -> pd.Series:
    """Network lens: rank by average correlation (centrality)."""
    corr = returns.corr().abs()
    centrality = corr.mean()
    return centrality.rank(ascending=False)


def lens_momentum(returns: pd.DataFrame) -> pd.Series:
    """Momentum lens: rank by absolute cumulative return."""
    cumret = (1 + returns).prod() - 1
    return cumret.abs().rank(ascending=False)


def lens_drawdown(prices: pd.DataFrame) -> pd.Series:
    """Drawdown lens: rank by current drawdown from peak."""
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    current_dd = drawdown.iloc[-1].abs()
    return current_dd.rank(ascending=False)


def lens_beta(returns: pd.DataFrame, market_col: str = None) -> pd.Series:
    """Beta lens: rank by sensitivity to market."""
    # Find market proxy (first column with 'sp' or 'spy' in name)
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


def lens_skewness(returns: pd.DataFrame) -> pd.Series:
    """Skewness lens: rank by tail risk (negative skew = danger)."""
    skew = returns.skew()
    # More negative = more important (crash risk)
    return (-skew).rank(ascending=False)


def lens_kurtosis(returns: pd.DataFrame) -> pd.Series:
    """Kurtosis lens: rank by fat tails."""
    kurt = returns.kurtosis()
    return kurt.rank(ascending=False)


def lens_autocorrelation(returns: pd.DataFrame) -> pd.Series:
    """Autocorrelation lens: rank by persistence."""
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


def lens_volume_correlation(returns: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional dispersion lens.
    High dispersion = disagreement in market.
    """
    # Use rolling std of cross-sectional mean as proxy
    cross_mean = returns.mean(axis=1)
    cross_std = returns.std(axis=1)
    
    # Rank by correlation with cross-sectional stress
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


def lens_regime_persistence(returns: pd.DataFrame) -> pd.Series:
    """
    Regime persistence lens: which indicators are "sticky" vs mean-reverting.
    Uses Hurst exponent approximation.
    """
    hurst_approx = {}
    
    for col in returns.columns:
        series = returns[col].dropna()
        if len(series) < 50:
            continue
        
        try:
            # Simple R/S approximation
            n = len(series)
            mean = series.mean()
            std = series.std()
            
            if std == 0:
                continue
            
            # Cumulative deviation from mean
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


# All available lenses
LENSES = {
    'pca': lens_pca_variance,
    'volatility': lens_volatility,
    'correlation': lens_correlation_centrality,
    'momentum': lens_momentum,
    'beta': lens_beta,
    'skewness': lens_skewness,
    'kurtosis': lens_kurtosis,
    'autocorrelation': lens_autocorrelation,
    'dispersion': lens_volume_correlation,
    'persistence': lens_regime_persistence,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_panel() -> pd.DataFrame:
    """Load the master panel from database."""
    db_path = DATA_DIR / "prism.db"
    
    if not db_path.exists():
        # Try local path
        db_path = Path.home() / "prism_data" / "prism.db"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    
    # Load market data
    market = pd.read_sql("""
        SELECT date, ticker, value 
        FROM market_prices 
        WHERE value IS NOT NULL
    """, conn)
    
    # Load economic data
    econ = pd.read_sql("""
        SELECT date, series_id, value 
        FROM econ_values 
        WHERE value IS NOT NULL
    """, conn)
    
    conn.close()
    
    # Pivot to wide format
    market_wide = market.pivot(index='date', columns='ticker', values='value')
    econ_wide = econ.pivot(index='date', columns='series_id', values='value')
    
    # Combine
    panel = pd.concat([market_wide, econ_wide], axis=1)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    
    # Filter to recent 15 years for speed
    cutoff = panel.index.max() - pd.Timedelta(days=15*365)
    panel = panel.loc[cutoff:]
    
    print(f"📥 Loaded panel: {panel.shape[0]} days × {panel.shape[1]} indicators")
    print(f"   Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    
    return panel


# =============================================================================
# ROLLING LENS ANALYSIS
# =============================================================================

def run_lens_rolling(
    panel: pd.DataFrame, 
    lens_func, 
    window: int = 63,
    step: int = 5
) -> pd.DataFrame:
    """
    Run a lens across rolling windows.
    Returns DataFrame of rankings over time.
    """
    returns = panel.pct_change().dropna(how='all')
    
    results = []
    dates = []
    
    for i in range(window, len(returns), step):
        window_returns = returns.iloc[i-window:i]
        
        # Some lenses need prices, not returns
        if lens_func == lens_drawdown:
            window_data = panel.iloc[i-window:i]
        else:
            window_data = window_returns
        
        try:
            rankings = lens_func(window_data)
            if len(rankings) > 0:
                results.append(rankings)
                dates.append(returns.index[i])
        except Exception as e:
            continue
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results, index=dates)
    return df


def compute_consensus_metric(lens_rankings: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Compute consensus metric: how much do lenses agree at each time point?
    
    High consensus = lenses ranking indicators similarly
    Low consensus = lenses disagree (normal times)
    
    Returns series of consensus scores over time.
    """
    # Get common dates
    all_dates = None
    for name, df in lens_rankings.items():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))
    
    if not all_dates:
        return pd.Series(dtype=float)
    
    all_dates = sorted(all_dates)
    
    consensus = []
    
    for date in all_dates:
        # Get rankings from each lens for this date
        rankings_list = []
        for name, df in lens_rankings.items():
            if date in df.index:
                rankings_list.append(df.loc[date].dropna())
        
        if len(rankings_list) < 3:
            consensus.append(np.nan)
            continue
        
        # Find common indicators
        common = set(rankings_list[0].index)
        for r in rankings_list[1:]:
            common = common.intersection(set(r.index))
        
        if len(common) < 5:
            consensus.append(np.nan)
            continue
        
        common = sorted(common)
        
        # Compute average pairwise Spearman correlation
        correlations = []
        for i in range(len(rankings_list)):
            for j in range(i+1, len(rankings_list)):
                r1 = rankings_list[i].loc[common]
                r2 = rankings_list[j].loc[common]
                try:
                    corr, _ = stats.spearmanr(r1, r2)
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    pass
        
        if correlations:
            consensus.append(np.mean(correlations))
        else:
            consensus.append(np.nan)
    
    return pd.Series(consensus, index=all_dates)


def find_consensus_events(consensus: pd.Series, threshold_percentile: float = 90) -> List[dict]:
    """
    Find periods of unusually high consensus (potential regime events).
    
    Returns list of event dicts with date, consensus value, duration.
    """
    threshold = np.nanpercentile(consensus.dropna(), threshold_percentile)
    
    # Find periods above threshold
    above = consensus > threshold
    
    events = []
    in_event = False
    event_start = None
    event_peak = 0
    event_peak_date = None
    
    for date, is_high in above.items():
        if is_high and not in_event:
            # Start of event
            in_event = True
            event_start = date
            event_peak = consensus[date]
            event_peak_date = date
        elif is_high and in_event:
            # Continuing event
            if consensus[date] > event_peak:
                event_peak = consensus[date]
                event_peak_date = date
        elif not is_high and in_event:
            # End of event
            in_event = False
            events.append({
                'start': event_start,
                'peak_date': event_peak_date,
                'peak_consensus': event_peak,
                'duration_days': (date - event_start).days,
            })
    
    # Handle event at end of series
    if in_event:
        events.append({
            'start': event_start,
            'peak_date': event_peak_date,
            'peak_consensus': event_peak,
            'duration_days': (consensus.index[-1] - event_start).days,
        })
    
    return events


def score_lens_on_events(
    lens_rankings: pd.DataFrame, 
    events: List[dict],
    lookback_days: int = 90
) -> dict:
    """
    Score a lens on how early it detected each event.
    
    For each event, look at the lookback period and see:
    - Did top indicators change significantly before the event?
    - How many days before the peak did rankings start shifting?
    
    Returns dict with:
    - avg_lead_time: average days of early warning
    - hit_rate: % of events where lens gave early signal
    - stability: how consistent is the lens (low noise)
    """
    if len(events) == 0 or len(lens_rankings) == 0:
        return {'avg_lead_time': 0, 'hit_rate': 0, 'stability': 0}
    
    lead_times = []
    hits = 0
    
    for event in events:
        peak_date = event['peak_date']
        
        # Get rankings in lookback window
        lookback_start = peak_date - timedelta(days=lookback_days)
        
        mask = (lens_rankings.index >= lookback_start) & (lens_rankings.index <= peak_date)
        window_rankings = lens_rankings.loc[mask]
        
        if len(window_rankings) < 5:
            continue
        
        # Check for ranking changes: compare first half to second half
        mid = len(window_rankings) // 2
        first_half = window_rankings.iloc[:mid].mean()
        second_half = window_rankings.iloc[mid:].mean()
        
        # Find indicators that changed rank significantly (moved into top 5)
        top_5_now = set(second_half.nsmallest(5).index)
        top_5_before = set(first_half.nsmallest(5).index)
        
        new_arrivals = top_5_now - top_5_before
        
        if len(new_arrivals) > 0:
            hits += 1
            
            # Find when the first new arrival entered top 5
            for idx in range(len(window_rankings)):
                row = window_rankings.iloc[idx]
                top_5_at_time = set(row.nsmallest(5).index)
                if len(new_arrivals.intersection(top_5_at_time)) > 0:
                    lead_time = (peak_date - window_rankings.index[idx]).days
                    lead_times.append(lead_time)
                    break
    
    # Calculate stability (inverse of ranking volatility)
    rank_changes = lens_rankings.diff().abs().mean().mean()
    stability = 1 / (1 + rank_changes) if rank_changes > 0 else 1
    
    return {
        'avg_lead_time': np.mean(lead_times) if lead_times else 0,
        'hit_rate': hits / len(events) if events else 0,
        'stability': stability,
        'events_detected': hits,
        'total_events': len(events),
    }


# =============================================================================
# MAIN CALIBRATION
# =============================================================================

def run_calibration():
    """Run the full lens calibration."""
    
    print("=" * 70)
    print("🔧 PRISM CALIBRATION ENGINE - LENS BENCHMARK")
    print("=" * 70)
    print()
    
    # Load data
    panel = load_panel()
    
    # Run each lens
    print("\n🔬 Running lenses across rolling windows...")
    print("   Window: 63 days, Step: 5 days")
    print()
    
    lens_rankings = {}
    
    for name, func in LENSES.items():
        print(f"   Running {name}...", end=" ", flush=True)
        try:
            rankings = run_lens_rolling(panel, func, window=63, step=5)
            if len(rankings) > 0:
                lens_rankings[name] = rankings
                print(f"✅ ({len(rankings)} windows)")
            else:
                print("⚠️ No results")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n   Successfully ran {len(lens_rankings)} lenses")
    
    # Compute consensus metric
    print("\n📊 Computing consensus metric...")
    consensus = compute_consensus_metric(lens_rankings)
    print(f"   Consensus range: {consensus.min():.3f} to {consensus.max():.3f}")
    print(f"   Mean: {consensus.mean():.3f}, Std: {consensus.std():.3f}")
    
    # Find consensus events (regime breaks)
    print("\n🎯 Finding consensus events (potential regime breaks)...")
    events = find_consensus_events(consensus, threshold_percentile=90)
    print(f"   Found {len(events)} high-consensus events:")
    
    for i, event in enumerate(events[:10], 1):
        print(f"   {i}. {event['peak_date'].date()} - Consensus: {event['peak_consensus']:.3f} ({event['duration_days']} days)")
    
    # Score each lens
    print("\n⚖️ Scoring lenses on early warning capability...")
    print()
    
    lens_scores = {}
    
    for name, rankings in lens_rankings.items():
        scores = score_lens_on_events(rankings, events, lookback_days=90)
        lens_scores[name] = scores
        
        print(f"   {name:20} | Lead: {scores['avg_lead_time']:5.1f}d | "
              f"Hits: {scores['events_detected']}/{scores['total_events']} ({scores['hit_rate']*100:4.0f}%) | "
              f"Stability: {scores['stability']:.3f}")
    
    # Compute optimal weights
    print("\n🎚️ Computing optimal lens weights...")
    
    # Weight = lead_time * hit_rate * stability (normalized)
    weights = {}
    for name, scores in lens_scores.items():
        raw_weight = (
            scores['avg_lead_time'] * 
            scores['hit_rate'] * 
            (1 + scores['stability'])
        )
        weights[name] = max(raw_weight, 0.1)  # Minimum weight
    
    # Normalize to sum to len(weights)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total * len(weights) for k, v in weights.items()}
    
    # Sort by weight
    weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
    
    print()
    for name, weight in weights.items():
        bar = "█" * int(weight * 5)
        print(f"   {name:20} | {weight:.2f} | {bar}")
    
    # Save results
    print("\n💾 Saving calibration results...")
    
    # Lens scores CSV
    scores_df = pd.DataFrame(lens_scores).T
    scores_df.to_csv(CALIBRATION_DIR / "lens_scores.csv")
    print(f"   ✅ {CALIBRATION_DIR / 'lens_scores.csv'}")
    
    # Lens weights JSON
    with open(CALIBRATION_DIR / "lens_weights.json", 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"   ✅ {CALIBRATION_DIR / 'lens_weights.json'}")
    
    # Consensus events CSV
    events_df = pd.DataFrame(events)
    events_df.to_csv(CALIBRATION_DIR / "consensus_events.csv", index=False)
    print(f"   ✅ {CALIBRATION_DIR / 'consensus_events.csv'}")
    
    # Consensus timeseries
    consensus.to_csv(CALIBRATION_DIR / "consensus_timeseries.csv")
    print(f"   ✅ {CALIBRATION_DIR / 'consensus_timeseries.csv'}")
    
    # Create visualization
    create_calibration_plots(consensus, events, lens_scores, weights)
    
    print("\n" + "=" * 70)
    print("✅ CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {CALIBRATION_DIR}")
    print("\nTop 3 lenses by weight:")
    for name, weight in list(weights.items())[:3]:
        print(f"   🥇 {name}: {weight:.2f}")
    
    print("\nNext step: Use these weights in the main PRISM engine")
    
    return weights, lens_scores, events


def create_calibration_plots(
    consensus: pd.Series, 
    events: List[dict], 
    lens_scores: dict,
    weights: dict
):
    """Create visualization of calibration results."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Consensus over time
    ax1 = axes[0, 0]
    ax1.plot(consensus.index, consensus.values, color='steelblue', linewidth=1)
    ax1.fill_between(consensus.index, 0, consensus.values, alpha=0.3)
    
    # Mark events
    threshold = np.nanpercentile(consensus.dropna(), 90)
    ax1.axhline(threshold, color='red', linestyle='--', alpha=0.5, label='90th percentile')
    
    for event in events[:10]:
        ax1.axvline(event['peak_date'], color='red', alpha=0.5, linewidth=1)
    
    ax1.set_title('Lens Consensus Over Time\n(Higher = More Agreement = Potential Regime Break)', fontsize=11)
    ax1.set_ylabel('Consensus')
    ax1.legend()
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 2. Lens weights bar chart
    ax2 = axes[0, 1]
    names = list(weights.keys())
    vals = list(weights.values())
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))
    
    bars = ax2.barh(names, vals, color=colors)
    ax2.set_xlabel('Weight')
    ax2.set_title('Optimal Lens Weights\n(Based on Early Warning Performance)', fontsize=11)
    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Lead time vs hit rate scatter
    ax3 = axes[1, 0]
    
    for name, scores in lens_scores.items():
        ax3.scatter(scores['hit_rate'] * 100, scores['avg_lead_time'], 
                   s=scores['stability'] * 500, alpha=0.6, label=name)
    
    ax3.set_xlabel('Hit Rate (%)')
    ax3.set_ylabel('Average Lead Time (days)')
    ax3.set_title('Lens Performance\n(Size = Stability)', fontsize=11)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    # 4. Events timeline
    ax4 = axes[1, 1]
    
    if events:
        event_dates = [e['peak_date'] for e in events]
        event_values = [e['peak_consensus'] for e in events]
        
        ax4.scatter(event_dates, event_values, s=100, c='red', alpha=0.7, zorder=5)
        
        for i, event in enumerate(events[:5]):
            ax4.annotate(f"{event['peak_date'].strftime('%Y-%m')}", 
                        (event['peak_date'], event['peak_consensus']),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8)
    
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Peak Consensus')
    ax4.set_title('Detected Regime Events', fontsize=11)
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    
    plot_path = CALIBRATION_DIR / "calibration_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ {plot_path}")


if __name__ == "__main__":
    run_calibration()
