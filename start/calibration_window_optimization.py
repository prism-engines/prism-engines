#!/usr/bin/env python3
"""
PRISM Calibration Engine - Step 3: Window Optimization
=======================================================

Finds the optimal analysis window for each indicator type.

Key insight: VIX might need a 21-day window (fast-moving),
while yield curve might need 126-day window (slow-moving).

What it does:
1. Test multiple window sizes per indicator
2. Score each window on signal clarity
3. Group indicators by optimal window
4. Output window configuration

Usage:
    python calibration_window_optimization.py
    
Output:
    calibration/window_scores.csv
    calibration/optimal_windows.json
    calibration/window_groups.json
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

CALIBRATION_DIR = OUTPUT_DIR / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

# Windows to test
WINDOWS = [5, 10, 21, 42, 63, 126, 252]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_panel() -> pd.DataFrame:
    """Load the master panel."""
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
    
    cutoff = panel.index.max() - pd.Timedelta(days=10*365)
    panel = panel.loc[cutoff:]
    
    print(f"📥 Loaded panel: {panel.shape[0]} days × {panel.shape[1]} indicators")
    
    return panel


def load_consensus_events() -> pd.DataFrame:
    """Load consensus events."""
    events_file = CALIBRATION_DIR / "consensus_events.csv"
    
    if events_file.exists():
        df = pd.read_csv(events_file)
        df['start'] = pd.to_datetime(df['start'])
        df['peak_date'] = pd.to_datetime(df['peak_date'])
        return df
    
    return pd.DataFrame()


# =============================================================================
# WINDOW SCORING
# =============================================================================

def compute_signal_clarity(
    indicator: pd.Series,
    window: int
) -> float:
    """
    Compute signal clarity for a given window size.
    
    Good window = clear trends, low noise relative to signal
    Bad window = choppy, hard to distinguish signal from noise
    
    Returns: clarity score (higher = better)
    """
    if len(indicator.dropna()) < window * 2:
        return 0
    
    # Rolling mean and std
    rolling_mean = indicator.rolling(window).mean()
    rolling_std = indicator.rolling(window).std()
    
    # Signal-to-noise ratio
    # Signal = change in rolling mean over window
    signal = rolling_mean.diff(window).abs()
    noise = rolling_std
    
    # Avoid division by zero
    snr = signal / (noise + 1e-10)
    
    # Clarity = mean SNR (higher = clearer signal)
    clarity = snr.mean()
    
    if np.isnan(clarity) or np.isinf(clarity):
        return 0
    
    return clarity


def compute_trend_persistence(
    indicator: pd.Series,
    window: int
) -> float:
    """
    Measure how persistent trends are at this window size.
    
    Good window = trends last multiple windows
    Bad window = trends reverse within window (too slow) or noise (too fast)
    """
    if len(indicator.dropna()) < window * 3:
        return 0
    
    # Rolling return over window
    rolling_return = indicator.pct_change(window)
    
    # Autocorrelation of rolling returns
    # High autocorr = trends persist
    try:
        autocorr = rolling_return.autocorr(lag=1)
        if np.isnan(autocorr):
            return 0
        return max(0, autocorr)
    except:
        return 0


def compute_event_detection_score(
    indicator: pd.Series,
    events: pd.DataFrame,
    window: int
) -> float:
    """
    Score how well this window detects regime events.
    
    Test: Does volatility spike before events at this window?
    """
    if events.empty or len(indicator.dropna()) < window * 2:
        return 0
    
    # Rolling volatility
    rolling_vol = indicator.pct_change().rolling(window).std()
    
    event_scores = []
    
    for _, event in events.iterrows():
        peak_date = event['peak_date']
        
        # Get volatility leading up to event
        lookback = peak_date - timedelta(days=window * 2)
        
        mask = (rolling_vol.index >= lookback) & (rolling_vol.index <= peak_date)
        vol_window = rolling_vol.loc[mask].dropna()
        
        if len(vol_window) < 10:
            continue
        
        # Did volatility increase before event?
        mid = len(vol_window) // 2
        first_half = vol_window.iloc[:mid].mean()
        second_half = vol_window.iloc[mid:].mean()
        
        if first_half > 0:
            increase = (second_half - first_half) / first_half
            event_scores.append(max(0, increase))
    
    return np.mean(event_scores) if event_scores else 0


def score_window_for_indicator(
    indicator: pd.Series,
    window: int,
    events: pd.DataFrame
) -> dict:
    """
    Compute overall score for this indicator at this window size.
    """
    clarity = compute_signal_clarity(indicator, window)
    persistence = compute_trend_persistence(indicator, window)
    event_score = compute_event_detection_score(indicator, events, window)
    
    # Combined score
    total = (
        clarity * 0.4 +
        persistence * 0.3 +
        event_score * 0.3
    )
    
    return {
        'clarity': clarity,
        'persistence': persistence,
        'event_detection': event_score,
        'total': total,
    }


# =============================================================================
# INDICATOR TYPE DETECTION
# =============================================================================

def detect_indicator_type(indicator: pd.Series) -> str:
    """
    Detect the type of indicator based on behavior.
    
    Types:
    - fast: High frequency, quick mean reversion (VIX, daily returns)
    - medium: Normal market behavior (prices, spreads)
    - slow: Low frequency, persistent trends (rates, economic data)
    """
    if len(indicator.dropna()) < 100:
        return 'unknown'
    
    returns = indicator.pct_change().dropna()
    
    # Autocorrelation at different lags
    try:
        ac1 = abs(returns.autocorr(lag=1))
        ac5 = abs(returns.autocorr(lag=5))
        ac20 = abs(returns.autocorr(lag=20))
    except:
        return 'medium'
    
    # Mean reversion speed
    # Fast = low autocorrelation (quick mean reversion)
    # Slow = high autocorrelation (persistent)
    
    if ac1 < 0.05 and ac5 < 0.03:
        return 'fast'
    elif ac20 > 0.1:
        return 'slow'
    else:
        return 'medium'


def get_recommended_windows(indicator_type: str) -> list:
    """Get recommended windows for indicator type."""
    recommendations = {
        'fast': [5, 10, 21],
        'medium': [21, 42, 63],
        'slow': [63, 126, 252],
        'unknown': [21, 63, 126],
    }
    return recommendations.get(indicator_type, [63])


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def run_window_optimization():
    """Run window optimization for all indicators."""
    
    print("=" * 70)
    print("🔧 PRISM CALIBRATION ENGINE - WINDOW OPTIMIZATION")
    print("=" * 70)
    print()
    
    # Load data
    panel = load_panel()
    events = load_consensus_events()
    
    if events.empty:
        print("⚠️ No events found - using signal clarity only")
    else:
        print(f"   Using {len(events)} events for scoring")
    
    # Score each indicator at each window
    print(f"\n📊 Testing {len(WINDOWS)} window sizes: {WINDOWS}")
    print()
    
    all_scores = []
    optimal_windows = {}
    indicator_types = {}
    
    for col in panel.columns:
        indicator = panel[col]
        
        # Detect type
        ind_type = detect_indicator_type(indicator)
        indicator_types[col] = ind_type
        
        # Score each window
        best_window = 63
        best_score = 0
        
        for window in WINDOWS:
            scores = score_window_for_indicator(indicator, window, events)
            
            all_scores.append({
                'indicator': col,
                'window': window,
                'type': ind_type,
                **scores
            })
            
            if scores['total'] > best_score:
                best_score = scores['total']
                best_window = window
        
        optimal_windows[col] = {
            'optimal_window': best_window,
            'score': best_score,
            'type': ind_type,
        }
    
    scores_df = pd.DataFrame(all_scores)
    
    print(f"   Tested {len(panel.columns)} indicators")
    
    # Analyze results
    print("\n📈 Window Analysis by Indicator Type:")
    
    type_windows = defaultdict(list)
    for ind, info in optimal_windows.items():
        type_windows[info['type']].append(info['optimal_window'])
    
    for ind_type, windows in type_windows.items():
        avg_window = np.mean(windows)
        mode_window = max(set(windows), key=windows.count)
        print(f"   {ind_type:10} | Avg: {avg_window:.0f}d | Mode: {mode_window}d | Count: {len(windows)}")
    
    # Group indicators by optimal window
    window_groups = defaultdict(list)
    for ind, info in optimal_windows.items():
        window_groups[info['optimal_window']].append(ind)
    
    print("\n🎯 Indicators by Optimal Window:")
    for window in sorted(window_groups.keys()):
        indicators = window_groups[window]
        print(f"\n   {window}-day window ({len(indicators)} indicators):")
        for ind in indicators[:5]:
            print(f"      • {ind}")
        if len(indicators) > 5:
            print(f"      ... and {len(indicators) - 5} more")
    
    # Create window configuration
    window_config = {
        'default': 63,
        'by_type': {
            'fast': 21,
            'medium': 63,
            'slow': 126,
        },
        'by_indicator': {ind: info['optimal_window'] 
                        for ind, info in optimal_windows.items()},
    }
    
    # Save results
    print("\n💾 Saving window optimization results...")
    
    scores_df.to_csv(CALIBRATION_DIR / "window_scores.csv", index=False)
    print(f"   ✅ {CALIBRATION_DIR / 'window_scores.csv'}")
    
    with open(CALIBRATION_DIR / "optimal_windows.json", 'w') as f:
        json.dump(optimal_windows, f, indent=2)
    print(f"   ✅ {CALIBRATION_DIR / 'optimal_windows.json'}")
    
    with open(CALIBRATION_DIR / "window_config.json", 'w') as f:
        json.dump(window_config, f, indent=2)
    print(f"   ✅ {CALIBRATION_DIR / 'window_config.json'}")
    
    # Create visualization
    create_window_plots(scores_df, optimal_windows, window_groups)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ WINDOW OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    print("\n📋 RECOMMENDED CONFIGURATION:")
    print(f"   Fast indicators (VIX, etc):     21-day window")
    print(f"   Medium indicators (prices):     63-day window")
    print(f"   Slow indicators (rates, econ):  126-day window")
    
    print("\n🏆 TOP INDICATORS BY WINDOW:")
    
    for window in [21, 63, 126]:
        if window in window_groups:
            top_at_window = sorted(
                [(ind, optimal_windows[ind]['score']) 
                 for ind in window_groups[window]],
                key=lambda x: x[1], reverse=True
            )[:3]
            
            print(f"\n   {window}-day:")
            for ind, score in top_at_window:
                print(f"      • {ind} (score: {score:.3f})")
    
    print(f"\n📁 Results saved to: {CALIBRATION_DIR}")
    
    return window_config, optimal_windows


def create_window_plots(scores_df, optimal_windows, window_groups):
    """Create window optimization visualization."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Average score by window size
    ax1 = axes[0, 0]
    
    avg_by_window = scores_df.groupby('window')['total'].mean()
    
    ax1.bar(avg_by_window.index.astype(str), avg_by_window.values, 
            color='steelblue', alpha=0.7)
    ax1.set_xlabel('Window Size (days)')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Average Score by Window Size', fontweight='bold')
    
    # Mark best overall
    best_window = avg_by_window.idxmax()
    ax1.axvline(list(avg_by_window.index.astype(str)).index(str(best_window)), 
                color='red', linestyle='--', alpha=0.7)
    
    # 2. Distribution of optimal windows
    ax2 = axes[0, 1]
    
    window_counts = {w: len(inds) for w, inds in window_groups.items()}
    
    ax2.bar([str(w) for w in sorted(window_counts.keys())],
            [window_counts[w] for w in sorted(window_counts.keys())],
            color='coral', alpha=0.7)
    ax2.set_xlabel('Optimal Window (days)')
    ax2.set_ylabel('Number of Indicators')
    ax2.set_title('Distribution of Optimal Windows', fontweight='bold')
    
    # 3. Score by indicator type and window
    ax3 = axes[1, 0]
    
    pivot = scores_df.pivot_table(
        index='type', columns='window', values='total', aggfunc='mean'
    )
    
    pivot.plot(kind='bar', ax=ax3, alpha=0.7)
    ax3.set_xlabel('Indicator Type')
    ax3.set_ylabel('Average Score')
    ax3.set_title('Score by Type and Window', fontweight='bold')
    ax3.legend(title='Window', bbox_to_anchor=(1.02, 1))
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    
    # 4. Top 15 indicators by optimal score
    ax4 = axes[1, 1]
    
    top_15 = sorted(optimal_windows.items(), 
                    key=lambda x: x[1]['score'], reverse=True)[:15]
    
    names = [x[0] for x in top_15]
    scores = [x[1]['score'] for x in top_15]
    windows = [x[1]['optimal_window'] for x in top_15]
    
    colors = plt.cm.viridis([w / 252 for w in windows])
    
    y_pos = range(len(names))
    bars = ax4.barh(y_pos, scores, color=colors)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(names)
    ax4.invert_yaxis()
    ax4.set_xlabel('Score')
    ax4.set_title('Top 15 Indicators\n(color = optimal window)', fontweight='bold')
    
    # Add window labels
    for i, (name, score, window) in enumerate(zip(names, scores, windows)):
        ax4.text(score + 0.01, i, f'{window}d', va='center', fontsize=8)
    
    plt.tight_layout()
    
    plot_path = CALIBRATION_DIR / "window_optimization.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ {plot_path}")


if __name__ == "__main__":
    run_window_optimization()
