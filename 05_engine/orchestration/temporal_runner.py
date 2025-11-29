#!/usr/bin/env python3
"""
PRISM Temporal Runner - Year-Based Windowed Analysis
=====================================================

Run all 14 PRISM lenses across discrete time windows to track how
indicator importance changes over different periods.

Usage (command line):
    python temporal_runner.py --start 2005 --end 2025 --increment 5
    python temporal_runner.py --increment 8
    python temporal_runner.py --start 2010 --end 2024 --increment 2 --parallel
    python temporal_runner.py --increment 5 --export-csv  # Also export CSV files

Usage (Python):
    from temporal_runner import TemporalRunner
    runner = TemporalRunner(start_year=2005, end_year=2025, increment=5)
    results = runner.run()

Output (in 06_output/temporal/):
    Primary:
        - prism_temporal.db : SQLite database with full results

    Optional (with --export-csv):
        - temporal_results_{increment}yr.csv : Long format with all windows
        - rank_evolution_{increment}yr.csv   : Pivot table (indicators × windows)
        - regime_stability_{increment}yr.csv : Spearman correlation between adjacent windows
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# LENS CONFIGURATION
# =============================================================================

ALL_LENSES = [
    'magnitude',
    'pca',
    'influence',
    'clustering',
    'decomposition',
    'granger',
    'dmd',
    'mutual_info',
    'network',
    'regime_switching',
    'anomaly',
    'transfer_entropy',
    'tda',
    'wavelet',
]


def load_lens_class(lens_name: str):
    """Dynamically load a lens class by name."""
    lens_map = {
        'magnitude': ('magnitude_lens', 'MagnitudeLens'),
        'pca': ('pca_lens', 'PCALens'),
        'influence': ('influence_lens', 'InfluenceLens'),
        'clustering': ('clustering_lens', 'ClusteringLens'),
        'decomposition': ('decomposition_lens', 'DecompositionLens'),
        'granger': ('granger_lens', 'GrangerLens'),
        'dmd': ('dmd_lens', 'DMDLens'),
        'mutual_info': ('mutual_info_lens', 'MutualInfoLens'),
        'network': ('network_lens', 'NetworkLens'),
        'regime_switching': ('regime_switching_lens', 'RegimeSwitchingLens'),
        'anomaly': ('anomaly_lens', 'AnomalyLens'),
        'transfer_entropy': ('transfer_entropy_lens', 'TransferEntropyLens'),
        'tda': ('tda_lens', 'TDALens'),
        'wavelet': ('wavelet_lens', 'WaveletLens'),
    }

    if lens_name not in lens_map:
        raise ValueError(f"Unknown lens: {lens_name}")

    module_name, class_name = lens_map[lens_name]

    try:
        from importlib import import_module
        module = import_module(f"05_engine.lenses.{module_name}")
        return getattr(module, class_name)
    except ImportError as e:
        # Try alternate import path
        try:
            lenses_path = PROJECT_ROOT / "05_engine" / "lenses"
            sys.path.insert(0, str(lenses_path))
            module = import_module(module_name)
            return getattr(module, class_name)
        except Exception:
            raise ImportError(f"Could not load lens {lens_name}: {e}")


def run_single_lens(args: Tuple[str, pd.DataFrame]) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Run a single lens on data. Used for parallel execution.

    Args:
        args: Tuple of (lens_name, dataframe)

    Returns:
        Tuple of (lens_name, ranking_dataframe or None)
    """
    lens_name, df = args
    try:
        LensClass = load_lens_class(lens_name)
        lens = LensClass()
        ranking = lens.rank_indicators(df)
        return (lens_name, ranking)
    except Exception as e:
        return (lens_name, None)


# =============================================================================
# TEMPORAL RUNNER CLASS
# =============================================================================

class TemporalRunner:
    """
    Run PRISM analysis across discrete time windows.

    Generates windows like: 2005-2010, 2010-2015, 2015-2020, etc.
    """

    def __init__(
        self,
        start_year: int = 2005,
        end_year: int = None,
        increment: int = 5,
        data_path: str = None,
        output_dir: str = None,
        lenses: List[str] = None,
        parallel: bool = False,
        max_workers: int = None,
        export_csv: bool = False,
    ):
        """
        Initialize temporal runner.

        Args:
            start_year: First year to include (default 2005)
            end_year: Last year to include (default: current year)
            increment: Years per window (default 5)
            data_path: Path to master_panel.csv (default: data/raw/master_panel.csv)
            output_dir: Output directory (default: 06_output/temporal/)
            lenses: List of lenses to run (default: all 14)
            parallel: Use parallel processing for lenses
            max_workers: Max parallel workers (default: CPU count)
            export_csv: Also export CSV files (default: False, SQLite only)
        """
        self.start_year = start_year
        self.end_year = end_year or datetime.now().year
        self.increment = increment
        self.parallel = parallel
        self.max_workers = max_workers
        self.export_csv = export_csv

        # Paths
        self.project_root = PROJECT_ROOT
        self.data_path = Path(data_path) if data_path else self.project_root / "data" / "raw" / "master_panel.csv"
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "06_output" / "temporal"

        # Lenses
        self.lenses = lenses or ALL_LENSES

        # Storage
        self.panel: Optional[pd.DataFrame] = None
        self.windows: List[Tuple[int, int]] = []
        self.results: Dict[str, Any] = {}
        self.db = None  # Database manager

    def generate_windows(self) -> List[Tuple[int, int]]:
        """Generate time windows based on increment."""
        windows = []
        current = self.start_year

        while current < self.end_year:
            window_end = min(current + self.increment, self.end_year)
            windows.append((current, window_end))
            current = window_end

        self.windows = windows
        return windows

    def load_data(self) -> pd.DataFrame:
        """Load and prepare master panel data."""
        print(f"Loading data from {self.data_path}...")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load with date parsing
        panel = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

        # Ensure datetime index
        if not isinstance(panel.index, pd.DatetimeIndex):
            panel.index = pd.to_datetime(panel.index)

        panel = panel.sort_index()

        print(f"  Loaded: {panel.shape[0]} rows, {panel.shape[1]} columns")
        print(f"  Date range: {panel.index[0].date()} to {panel.index[-1].date()}")

        self.panel = panel
        return panel

    def slice_by_window(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Extract data for a specific time window."""
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        window_data = self.panel.loc[start_date:end_date].copy()

        # Clean: forward fill, backward fill, drop remaining NaN columns
        window_data = window_data.ffill().bfill()

        # Drop columns that are all NaN in this window
        window_data = window_data.dropna(axis=1, how='all')

        # Drop rows with any remaining NaN
        window_data = window_data.dropna()

        return window_data

    def prepare_for_lens(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame in format expected by lenses (with 'date' column)."""
        result = df.reset_index()
        result.columns = ['date'] + list(result.columns[1:])
        return result

    def run_lenses_on_window(
        self,
        window_data: pd.DataFrame,
        window_label: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all lenses on a single window.

        Args:
            window_data: Data for this window
            window_label: Label like "2005-2010"

        Returns:
            Dict mapping lens_name -> ranking DataFrame
        """
        prepared = self.prepare_for_lens(window_data)
        rankings = {}

        if self.parallel:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(run_single_lens, (lens, prepared)): lens
                    for lens in self.lenses
                }

                for future in as_completed(futures):
                    lens_name, ranking = future.result()
                    if ranking is not None:
                        rankings[lens_name] = ranking
        else:
            # Sequential execution
            for lens_name in self.lenses:
                try:
                    LensClass = load_lens_class(lens_name)
                    lens = LensClass()
                    ranking = lens.rank_indicators(prepared)
                    rankings[lens_name] = ranking
                except Exception as e:
                    print(f"    Warning: {lens_name} failed: {e}")

        return rankings

    def compute_consensus(self, rankings: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute consensus ranking from multiple lens rankings.

        Uses Borda count: each indicator gets points = n - rank
        """
        if not rankings:
            return pd.DataFrame()

        # Collect all indicators
        all_indicators = set()
        for ranking_df in rankings.values():
            if 'indicator' in ranking_df.columns:
                all_indicators.update(ranking_df['indicator'].tolist())

        n = len(all_indicators)
        scores = {ind: 0 for ind in all_indicators}
        lens_count = {ind: 0 for ind in all_indicators}

        # Borda count scoring
        for lens_name, ranking_df in rankings.items():
            if 'indicator' not in ranking_df.columns or 'rank' not in ranking_df.columns:
                continue

            for _, row in ranking_df.iterrows():
                ind = row['indicator']
                rank = row['rank']
                scores[ind] += (n - rank)
                lens_count[ind] += 1

        # Build consensus DataFrame
        consensus = []
        for ind in all_indicators:
            avg_score = scores[ind] / lens_count[ind] if lens_count[ind] > 0 else 0
            consensus.append({
                'indicator': ind,
                'consensus_score': avg_score,
                'n_lenses': lens_count[ind],
            })

        consensus_df = pd.DataFrame(consensus)
        consensus_df = consensus_df.sort_values('consensus_score', ascending=False)
        consensus_df['consensus_rank'] = range(1, len(consensus_df) + 1)

        return consensus_df.reset_index(drop=True)

    def compute_regime_stability(
        self,
        window_rankings: Dict[str, pd.DataFrame],
        windows: List[Tuple[int, int]]
    ) -> pd.DataFrame:
        """
        Compute Spearman correlation between adjacent time windows.

        This measures how stable the indicator rankings are across regime transitions.
        Dips in correlation indicate potential regime changes in market structure.

        Args:
            window_rankings: Dict mapping window_label -> consensus DataFrame
            windows: List of (start_year, end_year) tuples

        Returns:
            DataFrame with columns: transition_year, window_from, window_to, spearman_corr, p_value
        """
        stability_scores = []

        # Get window labels in order
        window_labels = [f"{start}-{end}" for start, end in windows]

        for i in range(len(window_labels) - 1):
            label_t0 = window_labels[i]
            label_t1 = window_labels[i + 1]

            if label_t0 not in window_rankings or label_t1 not in window_rankings:
                continue

            consensus_t0 = window_rankings[label_t0]
            consensus_t1 = window_rankings[label_t1]

            # Get common indicators
            indicators_t0 = set(consensus_t0['indicator'].tolist())
            indicators_t1 = set(consensus_t1['indicator'].tolist())
            common_indicators = indicators_t0 & indicators_t1

            if len(common_indicators) < 5:
                continue

            # Build aligned rank vectors
            ranks_t0 = consensus_t0.set_index('indicator').loc[list(common_indicators), 'consensus_rank']
            ranks_t1 = consensus_t1.set_index('indicator').loc[list(common_indicators), 'consensus_rank']

            # Compute Spearman correlation
            corr, p_value = spearmanr(ranks_t0.values, ranks_t1.values)

            # Transition year is the end of first window / start of second
            transition_year = windows[i][1]

            stability_scores.append({
                'transition_year': transition_year,
                'window_from': label_t0,
                'window_to': label_t1,
                'spearman_corr': corr,
                'p_value': p_value,
                'n_indicators': len(common_indicators),
            })

        return pd.DataFrame(stability_scores)

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the full temporal analysis.

        Returns:
            Dictionary with all results
        """
        start_time = datetime.now()

        if verbose:
            print("=" * 60)
            print("PRISM TEMPORAL ANALYSIS")
            print("=" * 60)
            print(f"Start year:  {self.start_year}")
            print(f"End year:    {self.end_year}")
            print(f"Increment:   {self.increment} years")
            print(f"Parallel:    {self.parallel}")
            print(f"Lenses:      {len(self.lenses)}")
            print()

        # Load data
        self.load_data()

        # Generate windows
        windows = self.generate_windows()
        if verbose:
            print(f"\nGenerated {len(windows)} time windows:")
            for start, end in windows:
                print(f"  {start}-{end}")
            print()

        # Process each window
        all_results = []
        window_rankings = {}
        lens_results_by_window = {}  # Store raw lens results for database

        for i, (start, end) in enumerate(windows):
            window_label = f"{start}-{end}"

            if verbose:
                print(f"[{i+1}/{len(windows)}] Processing {window_label}...")

            # Slice data
            window_data = self.slice_by_window(start, end)

            if len(window_data) < 100:
                if verbose:
                    print(f"    Skipping: insufficient data ({len(window_data)} rows)")
                continue

            if verbose:
                print(f"    Data: {len(window_data)} rows, {len(window_data.columns)} indicators")

            # Run lenses
            rankings = self.run_lenses_on_window(window_data, window_label)

            if verbose:
                print(f"    Lenses completed: {len(rankings)}/{len(self.lenses)}")

            # Compute consensus
            consensus = self.compute_consensus(rankings)

            # Store results
            window_rankings[window_label] = consensus
            lens_results_by_window[window_label] = rankings  # Store raw lens results

            # Add to long-format results
            for _, row in consensus.iterrows():
                all_results.append({
                    'window': window_label,
                    'start_year': start,
                    'end_year': end,
                    'indicator': row['indicator'],
                    'consensus_rank': row['consensus_rank'],
                    'consensus_score': row['consensus_score'],
                    'n_lenses': row['n_lenses'],
                })

        # Build output DataFrames
        results_df = pd.DataFrame(all_results)

        # Pivot for rank evolution (indicators × windows)
        if not results_df.empty:
            rank_evolution = results_df.pivot(
                index='indicator',
                columns='window',
                values='consensus_rank'
            )
            # Sort by average rank
            rank_evolution['avg_rank'] = rank_evolution.mean(axis=1)
            rank_evolution = rank_evolution.sort_values('avg_rank')
        else:
            rank_evolution = pd.DataFrame()

        # Compute regime stability (Spearman correlation between adjacent windows)
        if verbose:
            print("\nComputing regime stability...")
        regime_stability = self.compute_regime_stability(window_rankings, windows)

        if verbose and not regime_stability.empty:
            print(f"  Computed {len(regime_stability)} transition correlations")

        # Store results
        self.results = {
            'long_format': results_df,
            'rank_evolution': rank_evolution,
            'regime_stability': regime_stability,
            'window_rankings': window_rankings,
            'lens_results_by_window': lens_results_by_window,  # Raw lens results
            'windows': windows,
            'metadata': {
                'start_year': self.start_year,
                'end_year': self.end_year,
                'increment': self.increment,
                'n_windows': len(windows),
                'lenses': self.lenses,
                'run_time': str(datetime.now() - start_time),
            }
        }

        # Save outputs
        self.save_results(verbose=verbose)

        if verbose:
            elapsed = datetime.now() - start_time
            print()
            print("=" * 60)
            print(f"COMPLETE - {elapsed}")
            print("=" * 60)

            if not rank_evolution.empty:
                print("\nTOP 10 INDICATORS (by average rank across all windows):")
                print(rank_evolution.head(10)[['avg_rank']].to_string())

        return self.results

    def save_results(self, verbose: bool = True):
        """Save results to SQLite database (primary) and optionally CSV files."""
        from utils.db_manager import TemporalDB

        self.output_dir.mkdir(parents=True, exist_ok=True)

        increment = self.increment

        # =================================================================
        # PRIMARY OUTPUT: SQLite Database
        # =================================================================
        db_path = self.output_dir / "prism_temporal.db"
        self.db = TemporalDB(str(db_path))
        self.db.init_schema()

        if verbose:
            print(f"\nSaving to database: {db_path}")

        # Insert windows and track IDs
        window_ids = {}
        for start, end in self.results['windows']:
            window_label = f"{start}-{end}"
            window_id = self.db.insert_window(start, end, increment)
            window_ids[window_label] = window_id

        # Insert consensus rankings for each window
        for window_label, consensus_df in self.results['window_rankings'].items():
            window_id = window_ids[window_label]
            count = self.db.insert_consensus(window_id, consensus_df)
            if verbose:
                print(f"  {window_label}: {count} consensus rankings")

        # Insert lens results for each window
        for window_label, lens_results in self.results['lens_results_by_window'].items():
            window_id = window_ids[window_label]
            count = self.db.insert_lens_results(window_id, lens_results)
            if verbose:
                print(f"  {window_label}: {count} lens results")

        # Insert regime stability
        if not self.results['regime_stability'].empty:
            count = self.db.insert_regime_stability(
                self.results['regime_stability'],
                window_ids
            )
            if verbose:
                print(f"  Regime stability: {count} transitions")

        if verbose:
            stats = self.db.get_stats()
            print(f"\nDatabase statistics:")
            for table, count in stats.items():
                print(f"  {table}: {count} rows")

        # =================================================================
        # OPTIONAL: CSV Export (for compatibility)
        # =================================================================
        if self.export_csv:
            if verbose:
                print(f"\nExporting CSV files...")

            # Long format
            long_path = self.output_dir / f"temporal_results_{increment}yr.csv"
            self.results['long_format'].to_csv(long_path, index=False)

            # Rank evolution pivot
            evolution_path = self.output_dir / f"rank_evolution_{increment}yr.csv"
            self.results['rank_evolution'].to_csv(evolution_path)

            # Regime stability
            stability_path = self.output_dir / f"regime_stability_{increment}yr.csv"
            self.results['regime_stability'].to_csv(stability_path, index=False)

            if verbose:
                print(f"  {long_path}")
                print(f"  {evolution_path}")
                print(f"  {stability_path}")

        if verbose:
            print(f"\nPrimary output: {db_path}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Temporal Analysis - Run lenses across time windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python temporal_runner.py --increment 5
  python temporal_runner.py --start 2005 --end 2025 --increment 5
  python temporal_runner.py --increment 8 --parallel
  python temporal_runner.py --increment 5 --export-csv
  python temporal_runner.py --start 2010 --end 2024 --increment 2 --lenses magnitude pca influence

Output:
  Primary: prism_temporal.db (SQLite database)
  Optional: CSV files (with --export-csv flag)
        """
    )

    parser.add_argument(
        '--start', '-s',
        type=int,
        default=2005,
        help='Start year (default: 2005)'
    )

    parser.add_argument(
        '--end', '-e',
        type=int,
        default=None,
        help='End year (default: current year)'
    )

    parser.add_argument(
        '--increment', '-i',
        type=int,
        default=5,
        help='Years per window (default: 5)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to master_panel.csv'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory'
    )

    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Use parallel processing for lenses'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Max parallel workers (default: CPU count)'
    )

    parser.add_argument(
        '--lenses', '-l',
        nargs='+',
        default=None,
        help='Specific lenses to run (default: all 14)'
    )

    parser.add_argument(
        '--list-lenses',
        action='store_true',
        help='List available lenses and exit'
    )

    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Also export CSV files (default: SQLite only)'
    )

    args = parser.parse_args()

    # List lenses option
    if args.list_lenses:
        print("Available lenses:")
        for lens in ALL_LENSES:
            print(f"  - {lens}")
        return

    # Create runner
    runner = TemporalRunner(
        start_year=args.start,
        end_year=args.end,
        increment=args.increment,
        data_path=args.data,
        output_dir=args.output,
        lenses=args.lenses,
        parallel=args.parallel,
        max_workers=args.workers,
        export_csv=args.export_csv,
    )

    # Run
    runner.run()


if __name__ == "__main__":
    main()
