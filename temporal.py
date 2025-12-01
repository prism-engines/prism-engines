"""
PRISM Engine - Temporal Analysis
================================

Runs ALL lenses across rolling time windows.

You don't choose which analysis to run. PRISM runs everything
and tells you what it found.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


# ============================================================
# LENS REGISTRY
# ============================================================

LENSES = {}

def lens(name: str):
    """Decorator to register a lens function."""
    def decorator(func):
        LENSES[name] = func
        return func
    return decorator


# ------------------------------------------------------------
# BUILT-IN LENSES
# ------------------------------------------------------------

@lens("correlation")
def lens_correlation(df: pd.DataFrame, config: dict) -> pd.Series:
    """Mean absolute correlation with all other indicators."""
    corr = df.corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    return corr.mean(axis=1).sort_values(ascending=False)


@lens("volatility")
def lens_volatility(df: pd.DataFrame, config: dict) -> pd.Series:
    """Rank by rolling standard deviation."""
    window = config.get("window", 60)
    vol = df.rolling(window).std().mean()
    return vol.sort_values(ascending=False)


@lens("pca")
def lens_pca(df: pd.DataFrame, config: dict) -> pd.Series:
    """Rank by first principal component loading."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    clean = df.dropna()
    if len(clean) < 10:
        return pd.Series(index=df.columns, data=range(len(df.columns)))
    
    scaled = StandardScaler().fit_transform(clean)
    pca = PCA(n_components=1)
    pca.fit(scaled)
    
    loadings = pd.Series(np.abs(pca.components_[0]), index=df.columns)
    return loadings.sort_values(ascending=False)


@lens("momentum")
def lens_momentum(df: pd.DataFrame, config: dict) -> pd.Series:
    """Rank by recent momentum (rate of change)."""
    window = config.get("window", 20)
    momentum = (df.iloc[-1] - df.iloc[-window]) / df.iloc[-window].abs()
    return momentum.abs().sort_values(ascending=False)


@lens("stability")
def lens_stability(df: pd.DataFrame, config: dict) -> pd.Series:
    """Rank by rolling correlation stability (low std = stable)."""
    window = config.get("window", 60)
    
    stability = {}
    mean_series = df.mean(axis=1)
    
    for col in df.columns:
        rolling_corr = df[col].rolling(window).corr(mean_series)
        stability[col] = 1.0 / (rolling_corr.std() + 0.01)  # Inverse std
    
    return pd.Series(stability).sort_values(ascending=False)


@lens("range")
def lens_range(df: pd.DataFrame, config: dict) -> pd.Series:
    """Rank by price range (max - min) / mean."""
    relative_range = (df.max() - df.min()) / df.mean().abs()
    return relative_range.sort_values(ascending=False)


# Placeholder lenses for your more complex ones
@lens("granger")
def lens_granger(df: pd.DataFrame, config: dict) -> pd.Series:
    """Placeholder for Granger causality lens."""
    # TODO: Import your implementation
    return pd.Series(index=df.columns, data=np.random.rand(len(df.columns)))


@lens("wavelet")
def lens_wavelet(df: pd.DataFrame, config: dict) -> pd.Series:
    """Placeholder for wavelet analysis lens."""
    # TODO: Import your implementation
    return pd.Series(index=df.columns, data=np.random.rand(len(df.columns)))


@lens("transfer_entropy")
def lens_transfer_entropy(df: pd.DataFrame, config: dict) -> pd.Series:
    """Placeholder for transfer entropy lens."""
    # TODO: Import your implementation
    return pd.Series(index=df.columns, data=np.random.rand(len(df.columns)))


@lens("dmd")
def lens_dmd(df: pd.DataFrame, config: dict) -> pd.Series:
    """Placeholder for Dynamic Mode Decomposition lens."""
    # TODO: Import your implementation
    return pd.Series(index=df.columns, data=np.random.rand(len(df.columns)))


@lens("clustering")
def lens_clustering(df: pd.DataFrame, config: dict) -> pd.Series:
    """Placeholder for clustering lens."""
    # TODO: Import your implementation
    return pd.Series(index=df.columns, data=np.random.rand(len(df.columns)))


# ============================================================
# TEMPORAL WINDOWING
# ============================================================

def run_temporal_analysis(
    data: pd.DataFrame,
    categories: Dict,
    n_windows: Optional[int] = None,
    window_days: int = 504,  # ~2 years of trading days
    step_days: int = 252,    # ~1 year step
    weight_method: str = "auto"
) -> Dict[str, Any]:
    """
    Run all lenses across rolling time windows.
    
    Args:
        data: Full DataFrame with DateTimeIndex
        categories: Category definitions with weights
        n_windows: Number of windows (None = auto)
        window_days: Size of each window in rows
        step_days: Step size between windows
        weight_method: Weighting method being used
    
    Returns:
        {
            'windows': List of window results,
            'rankings': Final consensus ranking,
            'lens_agreement': Which lenses agreed most
        }
    """
    
    # Get relevant columns
    all_indicators = []
    for cat_info in categories.values():
        all_indicators.extend(cat_info["indicators"])
    
    available = [col for col in all_indicators if col in data.columns]
    df = data[available].dropna()
    
    # Calculate windows
    total_rows = len(df)
    
    if n_windows is not None:
        # Fixed number of windows
        step_days = max(1, (total_rows - window_days) // (n_windows - 1)) if n_windows > 1 else total_rows
    
    windows = []
    all_rankings = []
    
    start_idx = 0
    window_num = 1
    
    while start_idx + window_days <= total_rows:
        end_idx = start_idx + window_days
        window_df = df.iloc[start_idx:end_idx]
        
        start_date = window_df.index[0]
        end_date = window_df.index[-1]
        
        print(f"  [{window_num}] {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Run all lenses
        lens_rankings = {}
        config = {"window": 60}
        
        for lens_name, lens_func in LENSES.items():
            try:
                ranking = lens_func(window_df, config)
                lens_rankings[lens_name] = ranking
            except Exception as e:
                print(f"      ✗ {lens_name}: {str(e)[:40]}")
        
        # Store window result
        windows.append({
            "start_date": start_date,
            "end_date": end_date,
            "n_rows": len(window_df),
            "lens_rankings": lens_rankings
        })
        
        # Aggregate rankings for this window
        window_consensus = aggregate_rankings(lens_rankings)
        window_consensus.name = end_date
        all_rankings.append(window_consensus)
        
        # Progress indicator
        n_lenses = len(lens_rankings)
        print(f"      {n_lenses} lenses complete")
        
        start_idx += step_days
        window_num += 1
        
        if n_windows and window_num > n_windows:
            break
    
    # Final consensus ranking (mean across all windows)
    if all_rankings:
        rankings_df = pd.DataFrame(all_rankings).T
        final_rankings = rankings_df.mean(axis=1).sort_values()
    else:
        final_rankings = pd.Series()
    
    # Lens agreement matrix
    lens_agreement = calculate_lens_agreement(windows)
    
    return {
        "windows": windows,
        "rankings": final_rankings,
        "lens_agreement": lens_agreement,
        "n_windows": len(windows),
        "lenses_used": list(LENSES.keys())
    }


def aggregate_rankings(lens_rankings: Dict[str, pd.Series]) -> pd.Series:
    """
    Aggregate rankings from multiple lenses into a single consensus.
    Uses mean rank across lenses.
    """
    if not lens_rankings:
        return pd.Series()
    
    # Convert to ranks (lower = better = more important)
    ranks = pd.DataFrame()
    for lens_name, ranking in lens_rankings.items():
        # Ranking series should already be sorted descending
        # Convert to rank positions
        ranks[lens_name] = ranking.rank(ascending=False)
    
    # Mean rank across lenses
    consensus = ranks.mean(axis=1).sort_values()
    
    return consensus


def calculate_lens_agreement(windows: List[Dict]) -> pd.DataFrame:
    """Calculate how much each pair of lenses agrees across all windows."""
    if not windows:
        return pd.DataFrame()
    
    # Use the last window for lens pair comparison
    last_window = windows[-1]
    lens_rankings = last_window.get("lens_rankings", {})
    
    if len(lens_rankings) < 2:
        return pd.DataFrame()
    
    # Build rank DataFrame
    ranks = pd.DataFrame()
    for lens_name, ranking in lens_rankings.items():
        ranks[lens_name] = ranking.rank()
    
    # Spearman correlation between lenses
    return ranks.corr(method="spearman")
