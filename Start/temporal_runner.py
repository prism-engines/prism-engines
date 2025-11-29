"""
PRISM Temporal Analysis Runner
==============================

Easy-to-use script for running temporal (rolling window) analysis
and generating visualizations of how indicator rankings change over time.

Usage:
    # Import and run
    from Start.temporal_runner import run_temporal_analysis, generate_all_plots

    results = run_temporal_analysis(panel_clean)
    generate_all_plots(results)

    # Or use quick_start for everything at once
    from Start.temporal_runner import quick_start
    results, summary = quick_start(panel_clean)

CLI Usage:
    python Start/temporal_runner.py

Performance Notes for 50-60+ Indicators:
----------------------------------------
- Default uses fast lenses (magnitude, pca, influence, clustering)
- For Chromebook: use StreamingTemporalPRISM for memory efficiency
- For Mac Mini: can use all lenses and finer step sizes
- Consider step_months=0.5 for higher resolution on powerful machines
"""

import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
PROJECT_ROOT = SCRIPT_DIR.parent

# Add to Python path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / '05_engine' / 'orchestration'))
sys.path.insert(0, str(PROJECT_ROOT / '08_visualization' / 'plotters'))

import pandas as pd
import numpy as np
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Performance profiles
PERFORMANCE_PROFILES = {
    'chromebook': {
        'lenses': ['magnitude', 'pca', 'influence'],  # Fast lenses only
        'step_months': 2.0,  # Larger steps = faster
        'window_years': 1.0,
        'streaming': True,  # Use streaming for memory
    },
    'standard': {
        'lenses': ['magnitude', 'pca', 'influence', 'clustering'],
        'step_months': 1.0,
        'window_years': 1.0,
        'streaming': False,
    },
    'powerful': {
        'lenses': ['magnitude', 'pca', 'influence', 'clustering', 'decomposition'],
        'step_months': 0.5,  # Higher resolution
        'window_years': 1.0,
        'streaming': False,
    },
}


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_temporal_analysis(
    panel: pd.DataFrame,
    profile: str = 'standard',
    window_years: float = None,
    step_months: float = None,
    lenses: list = None,
    verbose: bool = True
) -> dict:
    """
    Run temporal PRISM analysis on your data.

    Args:
        panel: Your cleaned data panel (datetime index, indicator columns)
        profile: Performance profile ('chromebook', 'standard', 'powerful')
        window_years: Override window size (default from profile)
        step_months: Override step size (default from profile)
        lenses: Override lenses to use
        verbose: Print progress

    Returns:
        Dictionary with temporal analysis results

    Example:
        results = run_temporal_analysis(panel_clean, profile='chromebook')
    """
    from temporal_analysis import TemporalPRISM, StreamingTemporalPRISM

    # Get profile settings
    config = PERFORMANCE_PROFILES.get(profile, PERFORMANCE_PROFILES['standard'])

    # Apply overrides
    window_years = window_years or config['window_years']
    step_months = step_months or config['step_months']
    lenses = lenses or config['lenses']

    window_days = int(window_years * 252)
    step_days = int(step_months * 21)

    if verbose:
        print("=" * 60)
        print("PRISM TEMPORAL ANALYSIS")
        print("=" * 60)
        print(f"Profile:     {profile}")
        print(f"Window:      {window_years} year(s) ({window_days} days)")
        print(f"Step:        {step_months} month(s) ({step_days} days)")
        print(f"Lenses:      {lenses}")
        print(f"Indicators:  {len(panel.columns)}")
        print(f"Date range:  {panel.index[0]} to {panel.index[-1]}")
        print()

    # Create analyzer
    temporal = TemporalPRISM(panel, lenses=lenses)

    # Progress callback
    def progress(current, total):
        if verbose and current % 5 == 0:
            pct = current / total * 100
            print(f"  Progress: {current}/{total} windows ({pct:.0f}%)")

    # Run analysis
    start_time = datetime.now()

    results = temporal.run_rolling_analysis(
        window_days=window_days,
        step_days=step_days,
        progress_callback=progress if verbose else None
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    if verbose:
        print()
        print(f"Completed in {elapsed:.1f} seconds")
        print(f"Analyzed {results['metadata']['n_windows']} time periods")

    return results


def generate_all_plots(
    results: dict,
    output_dir: str = None,
    show_plots: bool = True
) -> dict:
    """
    Generate all temporal visualizations.

    Args:
        results: Output from run_temporal_analysis()
        output_dir: Directory to save plots (if None, just displays)
        show_plots: Whether to display plots interactively

    Returns:
        Dictionary mapping plot type -> file path (if saved)
    """
    from temporal_plots import (
        plot_ranking_evolution,
        plot_bump_chart,
        plot_rank_heatmap,
        plot_rank_changes,
        plot_stability_analysis,
    )

    print("\nGenerating visualizations...")

    saved_files = {}

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # 1. Ranking evolution line chart
    print("  1. Ranking evolution...")
    save_path = str(Path(output_dir) / 'ranking_evolution.png') if output_dir else None
    plot_ranking_evolution(results, save_path=save_path, top_n=10)
    if save_path:
        saved_files['ranking_evolution'] = save_path

    # 2. Bump chart
    print("  2. Bump chart...")
    save_path = str(Path(output_dir) / 'bump_chart.png') if output_dir else None
    plot_bump_chart(results, save_path=save_path, top_n=12)
    if save_path:
        saved_files['bump_chart'] = save_path

    # 3. Rank heatmap
    print("  3. Rank heatmap...")
    save_path = str(Path(output_dir) / 'rank_heatmap.png') if output_dir else None
    plot_rank_heatmap(results, save_path=save_path, top_n=20)
    if save_path:
        saved_files['rank_heatmap'] = save_path

    # 4. Recent rank changes
    print("  4. Rank changes...")
    save_path = str(Path(output_dir) / 'rank_changes.png') if output_dir else None
    plot_rank_changes(results, save_path=save_path)
    if save_path:
        saved_files['rank_changes'] = save_path

    # 5. Stability analysis
    print("  5. Stability analysis...")
    save_path = str(Path(output_dir) / 'stability.png') if output_dir else None
    plot_stability_analysis(results, save_path=save_path)
    if save_path:
        saved_files['stability'] = save_path

    print("\nDone!")
    if output_dir:
        print(f"Saved to: {output_dir}")

    return saved_files


def get_summary(results: dict) -> pd.DataFrame:
    """
    Get a summary DataFrame of the temporal analysis.

    Returns DataFrame with columns:
    - indicator: Indicator name
    - current_rank: Most recent rank
    - avg_rank: Average rank over time
    - best_rank: Best (lowest) rank achieved
    - worst_rank: Worst (highest) rank
    - rank_change: Recent change (positive = improving)
    - stability: Consistency score
    """
    from temporal_analysis import TemporalPRISM

    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])

    summary = pd.DataFrame({
        'current_rank': rank_df.iloc[-1],
        'avg_rank': rank_df.mean(),
        'best_rank': rank_df.min(),
        'worst_rank': rank_df.max(),
        'rank_std': rank_df.std(),
    })

    # Recent change (last 6 periods)
    lookback = min(6, len(rank_df))
    recent = rank_df.iloc[-lookback:]
    summary['rank_change'] = recent.iloc[0] - recent.iloc[-1]

    # Stability
    summary['stability'] = 1 / (1 + summary['rank_std'])

    # Sort by current rank
    summary = summary.sort_values('current_rank')

    return summary.round(2)


def find_trending(results: dict, lookback: int = 6) -> dict:
    """
    Find indicators that are rising or falling in importance.

    Args:
        results: Output from run_temporal_analysis()
        lookback: Number of periods to look back

    Returns:
        Dictionary with 'rising' and 'falling' indicators
    """
    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])

    lookback = min(lookback, len(rank_df))
    recent = rank_df.iloc[-lookback:]

    change = recent.iloc[0] - recent.iloc[-1]

    rising = change[change > 2].sort_values(ascending=False)
    falling = change[change < -2].sort_values()

    return {
        'rising': rising.to_dict(),
        'falling': falling.to_dict(),
        'period': f"{recent.index[0].strftime('%Y-%m')} to {recent.index[-1].strftime('%Y-%m')}"
    }


# =============================================================================
# COLAB-FRIENDLY QUICK START
# =============================================================================

def quick_start(panel: pd.DataFrame) -> tuple:
    """
    One-liner to run everything.

    Usage:
        results, summary = quick_start(panel_clean)

    Returns:
        (results dict, summary DataFrame)
    """
    print("Running quick temporal analysis...")
    print()

    results = run_temporal_analysis(panel, profile='standard')
    summary = get_summary(results)

    print("\n" + "=" * 60)
    print("TOP 10 CURRENT RANKINGS")
    print("=" * 60)
    print(summary.head(10).to_string())

    trending = find_trending(results)
    if trending['rising']:
        print("\n📈 RISING:")
        for ind, change in list(trending['rising'].items())[:5]:
            print(f"   {ind}: +{change:.1f} ranks")

    if trending['falling']:
        print("\n📉 FALLING:")
        for ind, change in list(trending['falling'].items())[:5]:
            print(f"   {ind}: {change:.1f} ranks")

    return results, summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("PRISM Temporal Analysis Runner")
    print("=" * 40)
    print()
    print("Usage in Colab/Python:")
    print("  1. Load your data into 'panel_clean'")
    print("  2. results = run_temporal_analysis(panel_clean)")
    print("  3. generate_all_plots(results)")
    print()
    print("Or use quick_start:")
    print("  results, summary = quick_start(panel_clean)")
    print()
    print("Available profiles:")
    for name, config in PERFORMANCE_PROFILES.items():
        print(f"  - {name}: {len(config['lenses'])} lenses, {config['step_months']}mo steps")
