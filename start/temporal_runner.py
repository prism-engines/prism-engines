#!/usr/bin/env python3
"""
Temporal Analysis Runner

Command-line wrapper around the PRISM temporal engine.

Examples:

  # Basic run with standard profile (default dates)
  python start/temporal_runner.py --profile standard

  # Chromebook-friendly run
  python start/temporal_runner.py --profile chromebook

  # Powerful run, finance-only, last 20 years
  python start/temporal_runner.py --profile powerful \
      --systems finance \
      --start 2005-01-01

  # Multi-domain run with explicit indicators
  python start/temporal_runner.py --profile standard \
      --systems finance climate \
      --indicators sp500 vix t10y2y

This script is SAFE to run multiple times - it only reads from the DB and
writes outputs under prism_output/temporal/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Setup paths
SCRIPT_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from engine.orchestration.temporal_analysis import (
    TEMPORAL_PROFILES,
    build_temporal_config,
    run_temporal_analysis_from_db,
    run_temporal_analysis_from_panel,
    load_temporal_panel,
)
from data.sql.db_path import get_db_path


def get_default_db_path() -> str:
    """Get the default database path using the unified resolver."""
    return get_db_path()


def _get_output_dir() -> Path:
    """Get the output directory for temporal analysis results."""
    out_dir = PROJECT_ROOT / "prism_output" / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PRISM temporal (multi-resolution) analysis from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--profile",
        choices=sorted(TEMPORAL_PROFILES.keys()),
        default="standard",
        help="Temporal profile to use (controls lenses and resolution).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, use as much history as available.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). If omitted, use latest available date.",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Optional list of system IDs (e.g. finance, climate, biology).",
    )
    parser.add_argument(
        "--indicators",
        nargs="*",
        default=None,
        help="Optional list of specific indicator IDs to include.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override database path. If omitted, use default resolver.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned configuration; do not run the engine.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose progress information.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entry point for temporal analysis runner.

    Args:
        argv: Command-line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args(argv)

    # Resolve DB path using the same logic as other scripts
    db_path = args.db_path or get_default_db_path()

    print("=" * 60)
    print("PRISM TEMPORAL ANALYSIS RUNNER")
    print("=" * 60)
    print(f"Profile:       {args.profile}")
    print(f"DB path:       {db_path}")
    print(f"Systems:       {args.systems or '[ALL]'}")
    print(f"Indicators:    {args.indicators or '[ALL]'}")
    print(f"Start date:    {args.start or '[auto]'}")
    print(f"End date:      {args.end or '[latest]'}")
    print("-" * 60)

    # Build configuration
    try:
        cfg = build_temporal_config(
            profile=args.profile,
            start_date=args.start,
            end_date=args.end,
            systems=args.systems,
            indicators=args.indicators,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if args.dry_run:
        print("Dry run only - no analysis executed.")
        print("Resolved config:")
        print(f"  step_months:   {cfg.step_months}")
        print(f"  window_months: {list(cfg.window_months)}")
        print(f"  lenses:        {list(cfg.lenses)}")
        print(f"  frequency:     {cfg.frequency}")
        print(f"  min_history:   {cfg.min_history_years} years")
        return 0

    # Run the temporal engine
    try:
        if args.verbose:
            print("\nLoading panel data...")

        result = run_temporal_analysis_from_db(db_path=db_path, config=cfg)

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Save outputs
    out_dir = _get_output_dir()
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    scores_path = out_dir / f"temporal_scores_{cfg.profile_name}_{timestamp}.csv"
    meta_path = out_dir / f"temporal_metadata_{cfg.profile_name}_{timestamp}.json"

    # Also save a "latest" version for easy access
    scores_latest = out_dir / f"temporal_scores_{cfg.profile_name}.csv"
    meta_latest = out_dir / f"temporal_metadata_{cfg.profile_name}.json"

    result.scores.to_csv(scores_path)
    result.scores.to_csv(scores_latest)
    pd.Series(result.metadata).to_json(meta_path, indent=2)
    pd.Series(result.metadata).to_json(meta_latest, indent=2)

    print("\nAnalysis complete.")
    if not result.scores.empty:
        print(f"  Dates:      {result.scores.index.min().date()} -> {result.scores.index.max().date()}")
    print(f"  Rows:       {result.scores.shape[0]}")
    print(f"  Columns:    {result.scores.shape[1]}")
    print(f"  Lenses:     {len(cfg.lenses)}")
    print(f"  Output CSV: {scores_latest}")
    print(f"  Metadata:   {meta_latest}")
    print("=" * 60)

    return 0


# =============================================================================
# Legacy API (for backward compatibility with existing imports)
# =============================================================================

# Performance profiles (legacy format)
PERFORMANCE_PROFILES = {
    'chromebook': {
        'lenses': ['magnitude', 'pca', 'influence'],
        'step_months': 2.0,
        'window_years': 1.0,
        'streaming': True,
    },
    'standard': {
        'lenses': ['magnitude', 'pca', 'influence', 'clustering'],
        'step_months': 1.0,
        'window_years': 1.0,
        'streaming': False,
    },
    'powerful': {
        'lenses': ['magnitude', 'pca', 'influence', 'clustering', 'decomposition'],
        'step_months': 0.5,
        'window_years': 1.0,
        'streaming': False,
    },
}


def run_temporal_analysis(
    panel: pd.DataFrame,
    profile: str = 'standard',
    window_years: float = None,
    step_months: float = None,
    lenses: list = None,
    verbose: bool = True
) -> dict:
    """
    Run temporal analysis on your data.

    This is the legacy API for backward compatibility.
    New code should use run_temporal_analysis_from_panel().

    Args:
        panel: Your cleaned data panel (datetime index, indicator columns)
        profile: Performance profile ('chromebook', 'standard', 'powerful')
        window_years: Override window size (default from profile)
        step_months: Override step size (default from profile)
        lenses: Override lenses to use
        verbose: Print progress

    Returns:
        Dictionary with temporal analysis results
    """
    # Get profile settings
    profile_config = PERFORMANCE_PROFILES.get(profile, PERFORMANCE_PROFILES['standard'])

    # Apply overrides
    window_years = window_years or profile_config['window_years']
    step_months = step_months or profile_config['step_months']
    lenses = lenses or profile_config['lenses']

    if verbose:
        print("=" * 60)
        print("TEMPORAL ANALYSIS")
        print("=" * 60)
        print(f"Profile:     {profile}")
        print(f"Window:      {window_years} year(s)")
        print(f"Step:        {step_months} month(s)")
        print(f"Lenses:      {lenses}")
        print(f"Indicators:  {len(panel.columns)}")
        if len(panel) > 0:
            print(f"Date range:  {panel.index[0]} to {panel.index[-1]}")
        print()

    # Build config and run
    cfg = build_temporal_config(
        profile=profile,
        step_months=step_months,
        lenses=lenses,
    )

    result = run_temporal_analysis_from_panel(panel, cfg)

    # Convert to legacy format
    return {
        'timestamps': list(result.scores.index),
        'rankings': {col: list(result.scores[col]) for col in result.scores.columns if '_rank' in col},
        'importance': {col: list(result.scores[col]) for col in result.scores.columns if '_score' in col},
        'metadata': result.metadata,
    }


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
    rankings = results.get('rankings', {})
    if not rankings:
        return pd.DataFrame()

    rank_df = pd.DataFrame(rankings, index=results.get('timestamps', []))

    summary = pd.DataFrame({
        'current_rank': rank_df.iloc[-1] if len(rank_df) > 0 else pd.Series(),
        'avg_rank': rank_df.mean(),
        'best_rank': rank_df.min(),
        'worst_rank': rank_df.max(),
        'rank_std': rank_df.std(),
    })

    # Recent change (last 6 periods)
    if len(rank_df) > 0:
        lookback = min(6, len(rank_df))
        recent = rank_df.iloc[-lookback:]
        summary['rank_change'] = recent.iloc[0] - recent.iloc[-1]
    else:
        summary['rank_change'] = 0

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
    rankings = results.get('rankings', {})
    if not rankings:
        return {'rising': {}, 'falling': {}, 'period': 'N/A'}

    rank_df = pd.DataFrame(rankings, index=results.get('timestamps', []))

    if len(rank_df) < 2:
        return {'rising': {}, 'falling': {}, 'period': 'N/A'}

    lookback = min(lookback, len(rank_df))
    recent = rank_df.iloc[-lookback:]

    change = recent.iloc[0] - recent.iloc[-1]

    rising = change[change > 2].sort_values(ascending=False)
    falling = change[change < -2].sort_values()

    period = f"{recent.index[0].strftime('%Y-%m')} to {recent.index[-1].strftime('%Y-%m')}"

    return {
        'rising': rising.to_dict(),
        'falling': falling.to_dict(),
        'period': period,
    }


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
    if not summary.empty:
        print(summary.head(10).to_string())
    else:
        print("No rankings available")

    trending = find_trending(results)
    if trending.get('rising'):
        print("\nRISING:")
        for ind, change in list(trending['rising'].items())[:5]:
            print(f"   {ind}: +{change:.1f} ranks")

    if trending.get('falling'):
        print("\nFALLING:")
        for ind, change in list(trending['falling'].items())[:5]:
            print(f"   {ind}: {change:.1f} ranks")

    return results, summary


if __name__ == "__main__":
    raise SystemExit(main())
