#!/usr/bin/env python3
"""
PRISM Calibration Loader
========================

Loads calibration results and applies them to the PRISM engine.

This is the bridge between the calibration meta-engine and the main analysis.

Usage:
    # In any script:
    from calibration_loader import get_lens_weights, get_weighted_consensus
    
    weights = get_lens_weights()
    # Returns: {'momentum': 1.34, 'beta': 1.14, ...}

Or run directly to see current calibration:
    python calibration_loader.py
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

import sys
if __name__ == "__main__":
    _script_dir = Path(__file__).parent.parent
    import os
    os.chdir(_script_dir)
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

from output_config import OUTPUT_DIR, DATA_DIR

CALIBRATION_DIR = OUTPUT_DIR / "calibration"


def get_lens_weights() -> dict:
    """
    Load calibrated lens weights.
    Returns dict of {lens_name: weight}
    
    If no calibration exists, returns equal weights.
    """
    weights_file = CALIBRATION_DIR / "lens_weights.json"
    
    if weights_file.exists():
        with open(weights_file) as f:
            return json.load(f)
    else:
        # Default equal weights
        print("⚠️ No calibration found - using equal weights")
        return {
            'pca': 1.0,
            'volatility': 1.0,
            'correlation': 1.0,
            'momentum': 1.0,
            'beta': 1.0,
            'skewness': 1.0,
            'kurtosis': 1.0,
            'autocorrelation': 1.0,
            'dispersion': 1.0,
            'persistence': 1.0,
        }


def get_lens_scores() -> pd.DataFrame:
    """Load detailed lens scores from calibration."""
    scores_file = CALIBRATION_DIR / "lens_scores.csv"
    
    if scores_file.exists():
        return pd.read_csv(scores_file, index_col=0)
    else:
        return pd.DataFrame()


def get_consensus_events() -> pd.DataFrame:
    """Load detected consensus events (regime breaks)."""
    events_file = CALIBRATION_DIR / "consensus_events.csv"
    
    if events_file.exists():
        df = pd.read_csv(events_file)
        df['start'] = pd.to_datetime(df['start'])
        df['peak_date'] = pd.to_datetime(df['peak_date'])
        return df
    else:
        return pd.DataFrame()


def get_weighted_consensus(lens_rankings: dict) -> pd.Series:
    """
    Compute weighted consensus from multiple lens rankings.
    
    Args:
        lens_rankings: dict of {lens_name: pd.Series of indicator rankings}
        
    Returns:
        pd.Series of weighted average rankings per indicator
    """
    weights = get_lens_weights()
    
    # Combine all rankings into DataFrame
    all_rankings = pd.DataFrame(lens_rankings)
    
    # Apply weights
    weighted_sum = pd.Series(0.0, index=all_rankings.index)
    total_weight = 0
    
    for lens_name in all_rankings.columns:
        w = weights.get(lens_name, 1.0)
        weighted_sum += all_rankings[lens_name].fillna(0) * w
        total_weight += w
    
    if total_weight > 0:
        weighted_avg = weighted_sum / total_weight
    else:
        weighted_avg = all_rankings.mean(axis=1)
    
    return weighted_avg.sort_values()


def get_top_lenses(n: int = 5) -> list:
    """Get the top N lenses by calibrated weight."""
    weights = get_lens_weights()
    sorted_lenses = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_lenses[:n]]


def should_use_lens(lens_name: str, min_weight: float = 0.5) -> bool:
    """Check if a lens should be used based on calibration."""
    weights = get_lens_weights()
    return weights.get(lens_name, 0) >= min_weight


def print_calibration_summary():
    """Print a summary of current calibration."""
    
    print("=" * 60)
    print("📊 PRISM CALIBRATION STATUS")
    print("=" * 60)
    
    weights = get_lens_weights()
    scores = get_lens_scores()
    events = get_consensus_events()
    
    print(f"\n📁 Calibration directory: {CALIBRATION_DIR}")
    print(f"   Files exist: {CALIBRATION_DIR.exists()}")
    
    print(f"\n🔬 Lens Weights:")
    for lens, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 5)
        status = "✅" if weight >= 1.0 else "⚠️" if weight >= 0.5 else "❌"
        print(f"   {status} {lens:20} {weight:.2f} {bar}")
    
    if not scores.empty:
        print(f"\n📈 Lens Performance:")
        print(scores.to_string())
    
    if not events.empty:
        print(f"\n🎯 Detected Regime Events: {len(events)}")
        print(f"   Most recent: {events['peak_date'].max()}")
        print(f"   Highest consensus: {events['peak_consensus'].max():.3f}")
    
    print("\n" + "=" * 60)
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    
    top_lenses = get_top_lenses(3)
    weak_lenses = [l for l, w in weights.items() if w < 0.5]
    
    print(f"\n   Use these lenses (highest weight):")
    for lens in top_lenses:
        print(f"      ✅ {lens}")
    
    if weak_lenses:
        print(f"\n   Consider dropping these lenses (low performance):")
        for lens in weak_lenses:
            print(f"      ❌ {lens}")
    
    print("\n" + "=" * 60)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_weighted_engine_config() -> dict:
    """
    Create a configuration dict for the PRISM engine
    that uses calibrated weights.
    """
    weights = get_lens_weights()
    scores = get_lens_scores()
    
    # Determine which lenses to use (weight >= 0.5)
    active_lenses = [l for l, w in weights.items() if w >= 0.5]
    
    # Get optimal parameters from scores
    config = {
        'lens_weights': weights,
        'active_lenses': active_lenses,
        'consensus_threshold': 0.2,  # Based on calibration findings
        'window_size': 63,
        'step_size': 5,
    }
    
    # If we have scores, extract more insights
    if not scores.empty:
        avg_lead_time = scores['avg_lead_time'].mean()
        config['expected_lead_time_days'] = avg_lead_time
    
    return config


def apply_weights_to_rankings(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply calibrated weights to a DataFrame of lens rankings.
    
    Args:
        rankings_df: DataFrame with columns = lens names, rows = indicators
        
    Returns:
        DataFrame with 'weighted_rank' column added
    """
    weights = get_lens_weights()
    
    # Calculate weighted average
    weighted_sum = pd.Series(0.0, index=rankings_df.index)
    total_weight = 0
    
    for col in rankings_df.columns:
        if col in weights:
            w = weights[col]
            weighted_sum += rankings_df[col].fillna(rankings_df[col].max()) * w
            total_weight += w
    
    if total_weight > 0:
        rankings_df['weighted_rank'] = weighted_sum / total_weight
    else:
        rankings_df['weighted_rank'] = rankings_df.mean(axis=1)
    
    # Re-rank based on weighted average
    rankings_df['final_rank'] = rankings_df['weighted_rank'].rank()
    
    return rankings_df.sort_values('final_rank')


if __name__ == "__main__":
    print_calibration_summary()
