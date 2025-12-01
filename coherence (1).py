"""
PRISM Engine - Coherence Index
==============================

The Coherence Index measures: DO THE LENSES AGREE?

High coherence = All lenses see similar rankings
Low coherence  = Lenses disagree (which is often HEALTHY - diversity)

The insight: It's not about what any one lens says.
             It's about when they ALL start saying the same thing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def calculate_coherence(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the Coherence Index from analysis results.
    
    Args:
        results: Output from run_temporal_analysis containing:
            - windows: List of window results
            - Each window has: lens_rankings (dict of lens -> ranking series)
    
    Returns:
        {
            'overall': float (0-1),
            'timeseries': pd.Series (coherence for each window),
            'trend': str ('rising', 'falling', 'stable'),
            'warning': str or None,
            'by_lens_pair': pd.DataFrame (which lenses agree most)
        }
    """
    windows = results.get("windows", [])
    
    if not windows:
        return {
            "overall": 0.0,
            "timeseries": pd.Series(),
            "trend": "unknown",
            "warning": "No analysis windows found"
        }
    
    # Calculate coherence for each window
    coherence_values = []
    window_dates = []
    
    for window in windows:
        lens_rankings = window.get("lens_rankings", {})
        
        if len(lens_rankings) < 2:
            coherence_values.append(np.nan)
        else:
            coh = window_coherence(lens_rankings)
            coherence_values.append(coh)
        
        window_dates.append(window.get("end_date", None))
    
    # Build timeseries
    timeseries = pd.Series(coherence_values, index=pd.to_datetime(window_dates))
    timeseries = timeseries.dropna()
    
    # Overall coherence (mean across windows)
    overall = timeseries.mean() if len(timeseries) > 0 else 0.0
    
    # Trend
    trend = calculate_trend(timeseries)
    
    # Warning detection
    warning = detect_warning(timeseries, overall)
    
    # Lens pair agreement (from most recent window)
    by_lens_pair = None
    if windows and "lens_rankings" in windows[-1]:
        by_lens_pair = lens_pair_agreement(windows[-1]["lens_rankings"])
    
    return {
        "overall": overall,
        "timeseries": timeseries,
        "trend": trend,
        "warning": warning,
        "by_lens_pair": by_lens_pair
    }


def window_coherence(lens_rankings: Dict[str, pd.Series]) -> float:
    """
    Calculate coherence for a single window.
    
    Method: Spearman rank correlation between all lens pairs.
    High correlation = lenses agree on rankings.
    """
    if len(lens_rankings) < 2:
        return np.nan
    
    # Convert to DataFrame of ranks
    ranks_df = pd.DataFrame()
    for lens_name, rankings in lens_rankings.items():
        # Handle both Series and dict inputs
        if isinstance(rankings, dict):
            rankings = pd.Series(rankings)
        ranks_df[lens_name] = rankings.rank()
    
    # Drop any indicators that aren't in all lenses
    ranks_df = ranks_df.dropna()
    
    if len(ranks_df) < 3:  # Need at least 3 indicators
        return np.nan
    
    # Pairwise Spearman correlations
    corr_matrix = ranks_df.corr(method="spearman")
    
    # Mean of off-diagonal (excluding self-correlation)
    n = len(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    coherence = corr_matrix.values[mask].mean()
    
    return coherence


def lens_pair_agreement(lens_rankings: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Calculate which lens pairs agree most/least.
    Useful for understanding lens behavior.
    """
    ranks_df = pd.DataFrame()
    for lens_name, rankings in lens_rankings.items():
        if isinstance(rankings, dict):
            rankings = pd.Series(rankings)
        ranks_df[lens_name] = rankings.rank()
    
    ranks_df = ranks_df.dropna()
    
    return ranks_df.corr(method="spearman")


def calculate_trend(timeseries: pd.Series) -> str:
    """Determine if coherence is rising, falling, or stable."""
    if len(timeseries) < 3:
        return "unknown"
    
    # Simple linear regression slope
    x = np.arange(len(timeseries))
    y = timeseries.values
    
    slope = np.polyfit(x, y, 1)[0]
    
    # Threshold for "meaningful" trend
    threshold = 0.01 * len(timeseries)  # Scale with data length
    
    if slope > threshold:
        return "rising"
    elif slope < -threshold:
        return "falling"
    else:
        return "stable"


def detect_warning(timeseries: pd.Series, overall: float) -> str:
    """
    Detect concerning patterns in coherence.
    
    Warnings:
    - Sudden spike (all lenses agree = herd behavior?)
    - Recent rise above historical norm
    - Extremely low coherence (data quality issue?)
    """
    if len(timeseries) < 3:
        return None
    
    recent = timeseries.iloc[-1]
    historical_mean = timeseries.iloc[:-1].mean()
    historical_std = timeseries.iloc[:-1].std()
    
    # Sudden spike (> 2 std above mean)
    if recent > historical_mean + 2 * historical_std:
        return f"Coherence spike: {recent:.3f} vs historical {historical_mean:.3f}"
    
    # Rising above 0.8 (very high agreement)
    if recent > 0.8 and historical_mean < 0.6:
        return "High coherence - lenses strongly aligned (check for regime shift)"
    
    # Extremely low
    if overall < 0.1:
        return "Very low coherence - check data quality or lens compatibility"
    
    return None


# ============================================================
# COHERENCE INTERPRETATION
# ============================================================

def interpret_coherence(coherence: float, context: str = "markets") -> str:
    """
    Human-readable interpretation of coherence value.
    
    The meaning depends on context:
    - Markets: High coherence often = warning (herd behavior)
    - Ecosystem: Low coherence often = healthy (diversity)
    """
    
    interpretations = {
        "markets": {
            (0.0, 0.3): "Low coherence - diverse signals, normal market conditions",
            (0.3, 0.5): "Moderate coherence - some alignment emerging",
            (0.5, 0.7): "Elevated coherence - notable agreement across lenses",
            (0.7, 0.9): "High coherence - strong alignment, potential regime signal",
            (0.9, 1.0): "Very high coherence - unusual agreement, investigate"
        },
        "ecosystem": {
            (0.0, 0.3): "Low coherence - healthy diversity in system response",
            (0.3, 0.5): "Moderate coherence - some common drivers",
            (0.5, 0.7): "Elevated coherence - system under shared stress",
            (0.7, 0.9): "High coherence - fragility warning, reduced resilience",
            (0.9, 1.0): "Critical coherence - monoculture risk, cascade possible"
        }
    }
    
    context_interp = interpretations.get(context, interpretations["markets"])
    
    for (low, high), text in context_interp.items():
        if low <= coherence < high:
            return text
    
    return "Coherence outside expected range"
