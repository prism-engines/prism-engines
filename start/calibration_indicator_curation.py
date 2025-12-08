#!/usr/bin/env python3
"""
PRISM Calibration Engine - Step 2: Indicator Curation
======================================================

Analyzes all indicators and separates signal from noise.

What it does:
1. Score each indicator on predictive value
2. Find redundant indicators (highly correlated pairs)
3. Identify noise (indicators that never matter)
4. Output a curated indicator list

The goal: If you have 1000 indicators, reduce to the 50-100 that matter.

Usage:
    python calibration_indicator_curation.py
    
Output:
    calibration/indicator_scores.csv
    calibration/indicator_tiers.json
    calibration/redundancy_map.csv
    calibration/curated_indicators.json
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
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from output_config import OUTPUT_DIR, DATA_DIR

CALIBRATION_DIR = OUTPUT_DIR / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_panel() -> pd.DataFrame:
    """Load the master panel using central runtime loader."""
    from panel.runtime_loader import load_calibrated_panel
    from datetime import datetime, timedelta

    try:
        # Load last 15 years
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=15*365)).strftime('%Y-%m-%d')

        panel = load_calibrated_panel(start_date=start_date, end_date=end_date)

        if not panel.empty:
            print(f"📥 Loaded panel: {panel.shape[0]} days × {panel.shape[1]} indicators")
            return panel

    except Exception as e:
        print(f"⚠️ Central loader failed: {e}")

    print("⚠️ No data found in database")
    return pd.DataFrame()


def load_consensus_events() -> pd.DataFrame:
    """Load consensus events from calibration step 1."""
    events_file = CALIBRATION_DIR / "consensus_events.csv"
    
    if events_file.exists():
        df = pd.read_csv(events_file)
        df['start'] = pd.to_datetime(df['start'])
        df['peak_date'] = pd.to_datetime(df['peak_date'])
        return df
    else:
        print("⚠️ No consensus events found - run calibration_lens_benchmark.py first")
        return pd.DataFrame()


# =============================================================================
# INDICATOR SCORING
# =============================================================================

def score_indicator_signal_strength(
    indicator_series: pd.Series,
    events: pd.DataFrame,
    lookback_days: int = 90
) -> dict:
    """
    Score an indicator on how well it signals regime events.
    
    Metrics:
    - event_reaction: Does it move significantly before/during events?
    - lead_time: How early does it react?
    - consistency: Does it react the same way each time?
    """
    if events.empty or len(indicator_series.dropna()) < 100:
        return {
            'event_reaction': 0,
            'avg_lead_time': 0,
            'consistency': 0,
            'data_quality': 0,
            'total_score': 0,
        }
    
    reactions = []
    lead_times = []
    
    for _, event in events.iterrows():
        peak_date = event['peak_date']
        
        # Get indicator values around event
        lookback_start = peak_date - timedelta(days=lookback_days)
        lookforward_end = peak_date + timedelta(days=30)
        
        mask = (indicator_series.index >= lookback_start) & (indicator_series.index <= lookforward_end)
        window = indicator_series.loc[mask].dropna()
        
        if len(window) < 20:
            continue
        
        # Find pre-event vs during-event behavior
        pre_event = window.loc[window.index < peak_date]
        during_event = window.loc[window.index >= peak_date]
        
        if len(pre_event) < 10 or len(during_event) < 5:
            continue
        
        # Measure volatility change (did it get more volatile before event?)
        pre_vol = pre_event.pct_change().std()
        during_vol = during_event.pct_change().std()
        
        if pre_vol > 0:
            vol_ratio = during_vol / pre_vol
            reactions.append(vol_ratio)
        
        # Measure trend change (did it start moving?)
        pre_trend = (pre_event.iloc[-1] - pre_event.iloc[0]) / pre_event.iloc[0] if pre_event.iloc[0] != 0 else 0
        
        # Find when significant move started
        returns = pre_event.pct_change().dropna()
        if len(returns) > 0:
            # Find first day with > 1 std move
            threshold = returns.std() * 1.5
            big_moves = returns.abs() > threshold
            
            if big_moves.any():
                first_move_date = big_moves.idxmax()
                lead_time = (peak_date - first_move_date).days
                lead_times.append(lead_time)
    
    # Calculate scores
    event_reaction = np.mean(reactions) if reactions else 0
    avg_lead_time = np.mean(lead_times) if lead_times else 0
    consistency = 1 - (np.std(reactions) / np.mean(reactions)) if reactions and np.mean(reactions) > 0 else 0
    
    # Data quality score
    data_coverage = len(indicator_series.dropna()) / len(indicator_series)
    data_quality = data_coverage
    
    # Combined score
    total_score = (
        event_reaction * 0.3 +
        (avg_lead_time / 90) * 0.3 +  # Normalize to 90 days max
        consistency * 0.2 +
        data_quality * 0.2
    )
    
    return {
        'event_reaction': event_reaction,
        'avg_lead_time': avg_lead_time,
        'consistency': max(0, consistency),
        'data_quality': data_quality,
        'total_score': total_score,
    }


def score_indicator_noise(returns: pd.Series) -> dict:
    """
    Score how "noisy" an indicator is.
    
    Low noise = good signal-to-noise ratio
    High noise = random walk, hard to extract signal
    """
    if len(returns.dropna()) < 50:
        return {'noise_score': 1.0, 'autocorr': 0, 'entropy': 1}
    
    clean = returns.dropna()
    
    # Autocorrelation (higher = more predictable = less noise)
    try:
        autocorr = abs(clean.autocorr(lag=1))
    except:
        autocorr = 0
    
    # Approximate entropy (higher = more random = more noise)
    # Using simple proxy: ratio of large moves to small moves
    threshold = clean.std()
    large_moves = (clean.abs() > threshold).sum() / len(clean)
    
    # Noise score (0 = clean signal, 1 = pure noise)
    noise_score = (1 - autocorr) * 0.5 + large_moves * 0.5
    
    return {
        'noise_score': noise_score,
        'autocorr': autocorr,
        'move_ratio': large_moves,
    }


# =============================================================================
# REDUNDANCY DETECTION
# =============================================================================

def find_redundant_pairs(panel: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Find pairs of indicators that are highly correlated (redundant).
    
    Returns DataFrame of redundant pairs with correlation values.
    """
    returns = panel.pct_change().dropna(how='all')
    
    # Compute correlation matrix
    corr = returns.corr()
    
    # Find pairs above threshold
    redundant = []
    
    for i, col1 in enumerate(corr.columns):
        for j, col2 in enumerate(corr.columns):
            if i >= j:
                continue
            
            c = corr.loc[col1, col2]
            if abs(c) >= threshold:
                redundant.append({
                    'indicator_1': col1,
                    'indicator_2': col2,
                    'correlation': c,
                })
    
    return pd.DataFrame(redundant).sort_values('correlation', ascending=False)


def cluster_indicators(panel: pd.DataFrame, n_clusters: int = None) -> dict:
    """
    Cluster indicators into groups of similar behavior.
    
    Returns dict mapping indicator -> cluster_id
    """
    returns = panel.pct_change().dropna(how='all')
    
    # Fill NaN with 0 for clustering
    returns_filled = returns.fillna(0)
    
    # Transpose so indicators are rows
    X = returns_filled.T.values
    
    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    if X.shape[0] < 3:
        return {col: 0 for col in returns.columns}
    
    # Hierarchical clustering
    try:
        Z = linkage(X, method='ward')
        
        if n_clusters is None:
            # Auto-determine clusters (aim for 5-10 groups)
            n_clusters = min(10, max(3, X.shape[0] // 5))
        
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        
        return {col: int(c) for col, c in zip(returns.columns, clusters)}
    except:
        return {col: 0 for col in returns.columns}


def pick_best_from_cluster(
    cluster_indicators: list,
    scores: pd.DataFrame
) -> str:
    """
    From a cluster of redundant indicators, pick the best one.
    """
    cluster_scores = scores.loc[scores.index.isin(cluster_indicators)]
    
    if cluster_scores.empty:
        return cluster_indicators[0]
    
    return cluster_scores['total_score'].idxmax()


# =============================================================================
# TIER CLASSIFICATION
# =============================================================================

def classify_into_tiers(scores: pd.DataFrame) -> dict:
    """
    Classify indicators into tiers based on scores.
    
    Tier 1: Core indicators (top 20%) - always include
    Tier 2: Supporting indicators (next 30%) - include if needed
    Tier 3: Marginal indicators (next 30%) - optional
    Tier 4: Noise (bottom 20%) - exclude
    """
    sorted_scores = scores.sort_values('total_score', ascending=False)
    n = len(sorted_scores)
    
    tiers = {}
    
    for i, (indicator, row) in enumerate(sorted_scores.iterrows()):
        pct = i / n
        
        if pct < 0.20:
            tier = 1
        elif pct < 0.50:
            tier = 2
        elif pct < 0.80:
            tier = 3
        else:
            tier = 4
        
        tiers[indicator] = {
            'tier': tier,
            'score': row['total_score'],
            'rank': i + 1,
        }
    
    return tiers


# =============================================================================
# MAIN CURATION
# =============================================================================

def run_curation():
    """Run the full indicator curation process."""
    
    print("=" * 70)
    print("🔧 PRISM CALIBRATION ENGINE - INDICATOR CURATION")
    print("=" * 70)
    print()
    
    # Load data
    panel = load_panel()
    events = load_consensus_events()
    
    if events.empty:
        print("❌ Cannot proceed without consensus events")
        print("   Run calibration_lens_benchmark.py first")
        return
    
    print(f"   Found {len(events)} consensus events for scoring")
    
    returns = panel.pct_change().dropna(how='all')
    
    # Score each indicator
    print("\n📊 Scoring indicators on signal strength...")
    
    signal_scores = {}
    noise_scores = {}
    
    for col in panel.columns:
        # Signal strength
        signal = score_indicator_signal_strength(panel[col], events)
        signal_scores[col] = signal
        
        # Noise level
        if col in returns.columns:
            noise = score_indicator_noise(returns[col])
            noise_scores[col] = noise
    
    # Combine into DataFrame
    signal_df = pd.DataFrame(signal_scores).T
    noise_df = pd.DataFrame(noise_scores).T
    
    scores = signal_df.join(noise_df, how='left')
    scores['noise_score'] = scores['noise_score'].fillna(1)
    
    # Adjust total score by noise
    scores['adjusted_score'] = scores['total_score'] * (1 - scores['noise_score'] * 0.3)
    scores['total_score'] = scores['adjusted_score']
    
    print(f"   Scored {len(scores)} indicators")
    
    # Find redundant pairs
    print("\n🔗 Finding redundant indicator pairs...")
    redundant = find_redundant_pairs(panel, threshold=0.85)
    print(f"   Found {len(redundant)} highly correlated pairs (>0.85)")
    
    if len(redundant) > 0:
        print("\n   Top redundant pairs:")
        for _, row in redundant.head(10).iterrows():
            print(f"      {row['indicator_1']:20} ↔ {row['indicator_2']:20} ({row['correlation']:.2f})")
    
    # Cluster indicators
    print("\n🎯 Clustering similar indicators...")
    clusters = cluster_indicators(panel)
    
    cluster_counts = defaultdict(list)
    for ind, c in clusters.items():
        cluster_counts[c].append(ind)
    
    print(f"   Found {len(cluster_counts)} clusters")
    
    # Classify into tiers
    print("\n📈 Classifying into tiers...")
    tiers = classify_into_tiers(scores)
    
    tier_counts = defaultdict(int)
    for ind, info in tiers.items():
        tier_counts[info['tier']] += 1
    
    print(f"   Tier 1 (Core):       {tier_counts[1]} indicators")
    print(f"   Tier 2 (Supporting): {tier_counts[2]} indicators")
    print(f"   Tier 3 (Marginal):   {tier_counts[3]} indicators")
    print(f"   Tier 4 (Noise):      {tier_counts[4]} indicators")
    
    # Create curated list
    print("\n✨ Creating curated indicator list...")
    
    curated = {
        'tier_1': [],
        'tier_2': [],
        'all_recommended': [],
        'excluded': [],
    }
    
    # For each cluster, pick the best indicator
    selected_from_clusters = set()
    
    for cluster_id, members in cluster_counts.items():
        if len(members) > 1:
            best = pick_best_from_cluster(members, scores)
            selected_from_clusters.add(best)
            
            # Mark others as redundant
            for m in members:
                if m != best:
                    if m in tiers:
                        tiers[m]['redundant_to'] = best
    
    for indicator, info in tiers.items():
        tier = info['tier']
        is_redundant = 'redundant_to' in info
        
        if tier == 1 and not is_redundant:
            curated['tier_1'].append(indicator)
            curated['all_recommended'].append(indicator)
        elif tier == 2 and not is_redundant:
            curated['tier_2'].append(indicator)
            curated['all_recommended'].append(indicator)
        elif tier == 4 or is_redundant:
            curated['excluded'].append(indicator)
    
    print(f"\n   Curated list: {len(curated['all_recommended'])} indicators")
    print(f"   Excluded:     {len(curated['excluded'])} indicators")
    
    # Save results
    print("\n💾 Saving curation results...")
    
    # Indicator scores
    scores.to_csv(CALIBRATION_DIR / "indicator_scores.csv")
    print(f"   ✅ {CALIBRATION_DIR / 'indicator_scores.csv'}")
    
    # Tiers
    with open(CALIBRATION_DIR / "indicator_tiers.json", 'w') as f:
        json.dump(tiers, f, indent=2, default=str)
    print(f"   ✅ {CALIBRATION_DIR / 'indicator_tiers.json'}")
    
    # Redundancy map
    redundant.to_csv(CALIBRATION_DIR / "redundancy_map.csv", index=False)
    print(f"   ✅ {CALIBRATION_DIR / 'redundancy_map.csv'}")
    
    # Curated list
    with open(CALIBRATION_DIR / "curated_indicators.json", 'w') as f:
        json.dump(curated, f, indent=2)
    print(f"   ✅ {CALIBRATION_DIR / 'curated_indicators.json'}")
    
    # Create visualization
    create_curation_plots(scores, tiers, redundant, clusters)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ INDICATOR CURATION COMPLETE")
    print("=" * 70)
    
    print("\n🏆 TIER 1 INDICATORS (Core - Always Use):")
    tier_1_sorted = sorted(curated['tier_1'], 
                          key=lambda x: tiers[x]['score'], reverse=True)
    for ind in tier_1_sorted[:15]:
        score = tiers[ind]['score']
        print(f"   ✅ {ind:25} (score: {score:.3f})")
    
    if len(tier_1_sorted) > 15:
        print(f"   ... and {len(tier_1_sorted) - 15} more")
    
    print("\n❌ EXCLUDED INDICATORS (Noise/Redundant):")
    for ind in curated['excluded'][:10]:
        reason = "redundant" if 'redundant_to' in tiers.get(ind, {}) else "noise"
        print(f"   ❌ {ind:25} ({reason})")
    
    if len(curated['excluded']) > 10:
        print(f"   ... and {len(curated['excluded']) - 10} more")
    
    print(f"\n📁 Results saved to: {CALIBRATION_DIR}")
    
    return curated, scores, tiers


def create_curation_plots(scores, tiers, redundant, clusters):
    """Create visualization of curation results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Score distribution by tier
    ax1 = axes[0, 0]
    
    tier_data = {1: [], 2: [], 3: [], 4: []}
    for ind, info in tiers.items():
        tier_data[info['tier']].append(info['score'])
    
    colors = ['#27AE60', '#F39C12', '#E67E22', '#E74C3C']
    labels = ['Tier 1\n(Core)', 'Tier 2\n(Support)', 'Tier 3\n(Marginal)', 'Tier 4\n(Noise)']
    
    bp = ax1.boxplot([tier_data[i] for i in [1, 2, 3, 4]], 
                     labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Score Distribution by Tier', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Top 20 indicators
    ax2 = axes[0, 1]
    
    top_20 = scores.nlargest(20, 'total_score')
    
    y_pos = range(len(top_20))
    colors2 = ['#27AE60' if tiers[ind]['tier'] == 1 else '#F39C12' 
               for ind in top_20.index]
    
    ax2.barh(y_pos, top_20['total_score'], color=colors2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_20.index)
    ax2.invert_yaxis()
    ax2.set_xlabel('Score')
    ax2.set_title('Top 20 Indicators by Score', fontweight='bold')
    
    # 3. Signal vs Noise scatter
    ax3 = axes[1, 0]
    
    ax3.scatter(scores['noise_score'], scores['event_reaction'], 
               alpha=0.6, c=scores['total_score'], cmap='RdYlGn')
    ax3.set_xlabel('Noise Score (lower = better)')
    ax3.set_ylabel('Event Reaction (higher = better)')
    ax3.set_title('Signal vs Noise', fontweight='bold')
    ax3.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(1.0, color='green', linestyle='--', alpha=0.5)
    
    # 4. Cluster sizes
    ax4 = axes[1, 1]
    
    cluster_sizes = defaultdict(int)
    for ind, c in clusters.items():
        cluster_sizes[c] += 1
    
    sizes = sorted(cluster_sizes.values(), reverse=True)
    ax4.bar(range(len(sizes)), sizes, color='steelblue', alpha=0.7)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Number of Indicators')
    ax4.set_title('Indicator Clusters (similar behavior)', fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = CALIBRATION_DIR / "curation_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ {plot_path}")


if __name__ == "__main__":
    run_curation()
