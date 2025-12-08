"""
Temporal Analysis Runner
==============================

Resolution-based temporal analysis with parallel processing.
Optimized for Mac Mini 16GB RAM.

Usage:
    # Python API
    from start.temporal_runner_parallel import run_temporal_analysis, run_parallel

    results = run_temporal_analysis(panel, resolution='monthly')
    results = run_parallel(panel)  # Parallel with all lenses

    # Quick start
    from start.temporal_runner_parallel import quick_start
    results, summary = quick_start(panel)

CLI Usage:
    python start/temporal_runner_parallel.py --resolution monthly --parallel
    python start/temporal_runner_parallel.py --resolution weekly --lenses all
    python start/temporal_runner_parallel.py --full  # All lenses, parallel
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'engine_core' / 'orchestration'))
sys.path.insert(0, str(PROJECT_ROOT / 'visualization' / 'plotters'))

import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION - Optimized for Mac Mini 16GB
# =============================================================================

# Hardware profile
HARDWARE = {
    'name': 'Mac Mini 16GB',
    'ram_gb': 16,
    'recommended_workers': 6,  # Leave headroom for system
    'max_workers': 8,
    'chunk_size': 10,  # Windows per chunk for memory efficiency
}

# Resolution presets
RESOLUTION_PRESETS = {
    'weekly': {
        'frequency': 'W-FRI',
        'window_days': 252,
        'step_days': 5,
        'lookback_default': '2Y',
        'description': 'Tactical - recent regime shifts',
    },
    'monthly': {
        'frequency': 'M',
        'window_days': 252,
        'step_days': 21,
        'lookback_default': '10Y',
        'description': 'Strategic - cycle positioning (DEFAULT)',
    },
    'quarterly': {
        'frequency': 'Q',
        'window_days': 504,
        'step_days': 63,
        'lookback_default': '30Y',
        'description': 'Structural - long-term patterns',
    },
    'fine': {
        'frequency': 'W-FRI',
        'window_days': 126,
        'step_days': 5,
        'lookback_default': '5Y',
        'description': 'High-resolution - detailed evolution',
    },
}

# All 14 lenses
ALL_LENSES = [
    'magnitude', 'pca', 'granger', 'dmd', 'influence', 'mutual_info',
    'clustering', 'decomposition', 'wavelet', 'network', 'regime',
    'anomaly', 'transfer_entropy', 'tda',
]

# Fast lenses (< 1 sec each)
FAST_LENSES = ['magnitude', 'pca', 'influence', 'clustering']

# Core lenses (good balance)
CORE_LENSES = ['magnitude', 'pca', 'influence', 'clustering', 'decomposition', 'regime']

# Heavy lenses (benefit most from parallelization)
HEAVY_LENSES = ['granger', 'transfer_entropy', 'tda', 'mutual_info']


@dataclass
class TemporalConfig:
    """Configuration for temporal analysis."""
    resolution: str = 'monthly'
    window_days: int = 252
    step_days: int = 21
    lenses: List[str] = field(default_factory=lambda: CORE_LENSES.copy())
    parallel: bool = True  # Default to parallel for Mac Mini
    workers: int = 6
    start_date: str = None
    end_date: str = None
    output_dir: str = 'output/temporal'
    
    @classmethod
    def from_resolution(cls, resolution: str = 'monthly', **overrides) -> 'TemporalConfig':
        """Create config from resolution preset."""
        preset = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS['monthly'])
        
        config = cls(
            resolution=resolution,
            window_days=preset['window_days'],
            step_days=preset['step_days'],
        )
        
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        
        return config
    
    @property
    def window_months(self) -> float:
        return self.window_days / 21
    
    @property
    def step_months(self) -> float:
        return self.step_days / 21


# =============================================================================
# PARALLEL PROCESSING ENGINE
# =============================================================================

def _run_lens_on_window(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel lens execution.
    Runs a single lens on a single window.
    """
    lens_name, window_data, window_idx, window_start, window_end = args
    
    try:
        # Import here to avoid pickle issues
        from engine_core.lenses import get_lens
        
        lens = get_lens(lens_name)
        result = lens.analyze(window_data)
        
        return {
            'lens': lens_name,
            'window_idx': window_idx,
            'window_start': window_start,
            'window_end': window_end,
            'result': result,
            'success': True,
        }
    except Exception as e:
        return {
            'lens': lens_name,
            'window_idx': window_idx,
            'window_start': window_start,
            'window_end': window_end,
            'error': str(e),
            'success': False,
        }


def _run_all_lenses_on_window(args: Tuple) -> List[Dict[str, Any]]:
    """
    Worker function: run all specified lenses on one window.
    More efficient than running each lens separately.
    """
    window_data, window_idx, window_start, window_end, lenses = args
    
    results = []
    
    try:
        from engine_core.lenses import get_lens
        
        for lens_name in lenses:
            try:
                lens = get_lens(lens_name)
                result = lens.analyze(window_data)
                results.append({
                    'lens': lens_name,
                    'window_idx': window_idx,
                    'window_start': window_start,
                    'window_end': window_end,
                    'result': result,
                    'success': True,
                })
            except Exception as e:
                results.append({
                    'lens': lens_name,
                    'window_idx': window_idx,
                    'error': str(e),
                    'success': False,
                })
    except Exception as e:
        results.append({
            'window_idx': window_idx,
            'error': f"Window processing failed: {e}",
            'success': False,
        })
    
    return results


class ParallelTemporalEngine:
    """
    Parallel temporal analysis engine optimized for Mac Mini 16GB.
    """
    
    def __init__(
        self,
        panel: pd.DataFrame,
        lenses: List[str] = None,
        workers: int = None
    ):
        self.panel = panel
        self.lenses = lenses or CORE_LENSES
        self.workers = workers or HARDWARE['recommended_workers']
        
        # Validate
        if not isinstance(panel.index, pd.DatetimeIndex):
            if 'date' in panel.columns:
                self.panel = panel.set_index('date')
            else:
                raise ValueError("Panel must have DatetimeIndex or 'date' column")
    
    def generate_windows(
        self,
        window_days: int,
        step_days: int
    ) -> List[Tuple[pd.DataFrame, int, str, str]]:
        """Generate all analysis windows."""
        windows = []
        
        start_idx = 0
        window_idx = 0
        
        while start_idx + window_days <= len(self.panel):
            end_idx = start_idx + window_days
            window_data = self.panel.iloc[start_idx:end_idx].copy()
            
            window_start = self.panel.index[start_idx].strftime('%Y-%m-%d')
            window_end = self.panel.index[end_idx - 1].strftime('%Y-%m-%d')
            
            windows.append((window_data, window_idx, window_start, window_end))
            
            start_idx += step_days
            window_idx += 1
        
        return windows
    
    def run_parallel(
        self,
        window_days: int = 252,
        step_days: int = 21,
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Run temporal analysis with parallel processing.
        
        Strategy: Parallelize across windows (each worker processes all lenses for one window).
        This is more memory-efficient than parallelizing across lenses.
        """
        windows = self.generate_windows(window_days, step_days)
        n_windows = len(windows)
        
        if n_windows == 0:
            raise ValueError("No valid windows generated. Check data length and window size.")
        
        # Prepare work items: (window_data, window_idx, start, end, lenses)
        work_items = [
            (w[0], w[1], w[2], w[3], self.lenses) 
            for w in windows
        ]
        
        # Results storage
        all_results = []
        completed = 0
        
        # Use ProcessPoolExecutor for CPU-bound work
        # ThreadPoolExecutor would be better for I/O-bound, but lenses are CPU-bound
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(_run_all_lenses_on_window, item): idx 
                for idx, item in enumerate(work_items)
            }
            
            for future in as_completed(futures):
                window_results = future.result()
                all_results.extend(window_results)
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, n_windows)
        
        # Organize results
        return self._organize_results(all_results, windows)
    
    def run_sequential(
        self,
        window_days: int = 252,
        step_days: int = 21,
        progress_callback: callable = None
    ) -> Dict[str, Any]:
        """
        Run temporal analysis sequentially (for comparison/debugging).
        """
        from engine_core.lenses import get_lens
        
        windows = self.generate_windows(window_days, step_days)
        n_windows = len(windows)
        
        all_results = []
        
        for idx, (window_data, window_idx, window_start, window_end) in enumerate(windows):
            for lens_name in self.lenses:
                try:
                    lens = get_lens(lens_name)
                    result = lens.analyze(window_data)
                    all_results.append({
                        'lens': lens_name,
                        'window_idx': window_idx,
                        'window_start': window_start,
                        'window_end': window_end,
                        'result': result,
                        'success': True,
                    })
                except Exception as e:
                    all_results.append({
                        'lens': lens_name,
                        'window_idx': window_idx,
                        'error': str(e),
                        'success': False,
                    })
            
            if progress_callback:
                progress_callback(idx + 1, n_windows)
        
        return self._organize_results(all_results, windows)
    
    def _organize_results(
        self,
        all_results: List[Dict],
        windows: List[Tuple]
    ) -> Dict[str, Any]:
        """Organize raw results into structured output."""
        
        # Build timestamps list
        timestamps = [pd.Timestamp(w[3]) for w in windows]  # Use window end dates
        
        # Build scores and rankings per lens
        scores_by_lens = {lens: {} for lens in self.lenses}
        rankings_by_lens = {lens: {} for lens in self.lenses}
        
        # Group results by window
        results_by_window = {}
        for r in all_results:
            if r.get('success', False):
                widx = r['window_idx']
                if widx not in results_by_window:
                    results_by_window[widx] = {}
                results_by_window[widx][r['lens']] = r['result']
        
        # Extract scores and compute rankings per window
        indicators = list(self.panel.columns)
        n_windows = len(windows)
        
        # Initialize score matrices
        for lens in self.lenses:
            scores_by_lens[lens] = {ind: [np.nan] * n_windows for ind in indicators}
        
        for widx, lens_results in results_by_window.items():
            for lens_name, result in lens_results.items():
                if 'scores' in result:
                    for ind, score in result['scores'].items():
                        if ind in scores_by_lens[lens_name]:
                            scores_by_lens[lens_name][ind][widx] = score
        
        # Compute rankings from scores
        for lens in self.lenses:
            for widx in range(n_windows):
                window_scores = {
                    ind: scores_by_lens[lens][ind][widx] 
                    for ind in indicators 
                    if not np.isnan(scores_by_lens[lens][ind][widx])
                }
                
                if window_scores:
                    sorted_inds = sorted(window_scores.keys(), key=lambda x: window_scores[x], reverse=True)
                    for rank, ind in enumerate(sorted_inds, 1):
                        if ind not in rankings_by_lens[lens]:
                            rankings_by_lens[lens][ind] = [np.nan] * n_windows
                        rankings_by_lens[lens][ind][widx] = rank
        
        # Compute consensus rankings (average across lenses)
        consensus_rankings = {ind: [np.nan] * n_windows for ind in indicators}
        
        for widx in range(n_windows):
            for ind in indicators:
                ranks = [
                    rankings_by_lens[lens].get(ind, [np.nan] * n_windows)[widx]
                    for lens in self.lenses
                ]
                valid_ranks = [r for r in ranks if not np.isnan(r)]
                if valid_ranks:
                    consensus_rankings[ind][widx] = np.mean(valid_ranks)
        
        # Success rate
        success_count = sum(1 for r in all_results if r.get('success', False))
        total_count = len(all_results)
        
        return {
            'timestamps': timestamps,
            'scores': scores_by_lens,
            'rankings': consensus_rankings,
            'rankings_by_lens': rankings_by_lens,
            'metadata': {
                'n_windows': n_windows,
                'n_lenses': len(self.lenses),
                'n_indicators': len(indicators),
                'lenses': self.lenses,
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'total_computations': total_count,
            }
        }


# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================

def run_temporal_analysis(
    panel: pd.DataFrame,
    resolution: str = 'monthly',
    window_days: int = None,
    step_days: int = None,
    lenses: List[str] = None,
    parallel: bool = True,
    workers: int = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run temporal analysis on your data.

    Args:
        panel: Cleaned data panel (datetime index, indicator columns)
        resolution: 'weekly', 'monthly', 'quarterly', 'fine'
        window_days: Override window size (trading days)
        step_days: Override step size (trading days)  
        lenses: Lenses to use (default: CORE_LENSES)
        parallel: Use parallel processing (default: True)
        workers: Parallel workers (default: 6 for Mac Mini)
        verbose: Print progress

    Returns:
        Dictionary with temporal analysis results
    """
    # Build config
    config = TemporalConfig.from_resolution(resolution)
    
    if window_days is not None:
        config.window_days = window_days
    if step_days is not None:
        config.step_days = step_days
    if lenses is not None:
        config.lenses = lenses
    if workers is not None:
        config.workers = workers
    
    config.parallel = parallel

    if verbose:
        print("=" * 60)
        print("TEMPORAL ANALYSIS")
        print("=" * 60)
        print(f"Hardware:     {HARDWARE['name']}")
        print(f"Resolution:   {config.resolution}")
        print(f"Window:       {config.window_days} days ({config.window_months:.1f} months)")
        print(f"Step:         {config.step_days} days ({config.step_months:.1f} months)")
        print(f"Lenses:       {len(config.lenses)}")
        print(f"Parallel:     {config.parallel} ({config.workers} workers)")
        print(f"Indicators:   {len(panel.columns)}")
        print(f"Date range:   {panel.index[0]} to {panel.index[-1]}")
        print("-" * 60)

    # Create engine
    engine = ParallelTemporalEngine(
        panel=panel,
        lenses=config.lenses,
        workers=config.workers
    )

    # Progress callback
    def progress(current, total):
        if verbose:
            pct = current / total * 100
            bar_len = 30
            filled = int(bar_len * current / total)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r  [{bar}] {current}/{total} ({pct:.0f}%)", end='', flush=True)

    # Run analysis
    start_time = datetime.now()

    if config.parallel:
        results = engine.run_parallel(
            window_days=config.window_days,
            step_days=config.step_days,
            progress_callback=progress if verbose else None
        )
    else:
        results = engine.run_sequential(
            window_days=config.window_days,
            step_days=config.step_days,
            progress_callback=progress if verbose else None
        )

    elapsed = (datetime.now() - start_time).total_seconds()

    if verbose:
        print()  # New line after progress bar
        print("-" * 60)
        print(f"Completed in {elapsed:.1f} seconds")
        print(f"Windows:      {results['metadata']['n_windows']}")
        print(f"Success rate: {results['metadata']['success_rate']:.1%}")
        if config.parallel and elapsed > 0:
            rate = results['metadata']['n_windows'] / elapsed
            print(f"Performance:  {rate:.1f} windows/second")

    # Add config to results
    results['config'] = {
        'resolution': config.resolution,
        'window_days': config.window_days,
        'step_days': config.step_days,
        'lenses': config.lenses,
        'parallel': config.parallel,
        'workers': config.workers,
        'elapsed_seconds': elapsed,
    }

    return results


def run_parallel(
    panel: pd.DataFrame,
    resolution: str = 'monthly',
    workers: int = None,
    lenses: List[str] = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for parallel analysis with all lenses.
    """
    return run_temporal_analysis(
        panel,
        resolution=resolution,
        lenses=lenses or ALL_LENSES,
        parallel=True,
        workers=workers or HARDWARE['recommended_workers'],
        verbose=verbose,
        **kwargs
    )


def run_fast(panel: pd.DataFrame, resolution: str = 'monthly', **kwargs) -> Dict[str, Any]:
    """Quick analysis with fast lenses only."""
    return run_temporal_analysis(panel, resolution=resolution, lenses=FAST_LENSES, **kwargs)


def run_full(panel: pd.DataFrame, resolution: str = 'monthly', **kwargs) -> Dict[str, Any]:
    """Complete analysis with all 14 lenses, parallel."""
    return run_temporal_analysis(panel, resolution=resolution, lenses=ALL_LENSES, parallel=True, **kwargs)


# =============================================================================
# SUMMARY & ANALYSIS
# =============================================================================

def get_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """Get summary DataFrame of temporal analysis."""
    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])

    summary = pd.DataFrame({
        'current_rank': rank_df.iloc[-1],
        'avg_rank': rank_df.mean(),
        'best_rank': rank_df.min(),
        'worst_rank': rank_df.max(),
        'rank_std': rank_df.std(),
    })

    lookback = min(6, len(rank_df))
    recent = rank_df.iloc[-lookback:]
    summary['rank_change'] = recent.iloc[0] - recent.iloc[-1]
    summary['stability'] = 1 / (1 + summary['rank_std'])

    return summary.sort_values('current_rank').round(2)


def find_trending(results: Dict[str, Any], lookback: int = 6) -> Dict[str, Any]:
    """Find rising and falling indicators."""
    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])
    lookback = min(lookback, len(rank_df))
    recent = rank_df.iloc[-lookback:]
    change = recent.iloc[0] - recent.iloc[-1]

    return {
        'rising': change[change > 2].sort_values(ascending=False).to_dict(),
        'falling': change[change < -2].sort_values().to_dict(),
        'period': f"{recent.index[0].strftime('%Y-%m')} to {recent.index[-1].strftime('%Y-%m')}"
    }


def quick_start(panel: pd.DataFrame, resolution: str = 'monthly') -> tuple:
    """One-liner to run everything with parallel processing."""
    print(f"Running {resolution} temporal analysis (parallel)...")
    print()

    results = run_parallel(panel, resolution=resolution)
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
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Temporal Analysis - Mac Mini Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimized for Mac Mini 16GB RAM with parallel processing.

Resolutions:
  weekly     1-week steps, tactical analysis
  monthly    1-month steps, strategic analysis (DEFAULT)
  quarterly  3-month steps, structural analysis
  fine       1-week steps, 6-month window, high-resolution

Examples:
  python start/temporal_runner.py                          # Monthly, parallel, core lenses
  python start/temporal_runner.py --full                   # All 14 lenses, parallel
  python start/temporal_runner.py --resolution weekly      # Weekly analysis
  python start/temporal_runner.py --fast                   # Fast lenses only
  python start/temporal_runner.py --workers 4 --lenses magnitude pca regime
        """
    )
    
    parser.add_argument('--resolution', '-r', choices=['weekly', 'monthly', 'quarterly', 'fine'],
                        default='monthly', help='Analysis resolution')
    parser.add_argument('--parallel', '-p', action='store_true', default=True,
                        help='Use parallel processing (default: True)')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--workers', '-w', type=int, default=6, help='Parallel workers (default: 6)')
    parser.add_argument('--lenses', '-l', nargs='+', help='Specific lenses (or: all, fast, core)')
    parser.add_argument('--full', action='store_true', help='All 14 lenses with parallel')
    parser.add_argument('--fast', action='store_true', help='Fast lenses only')
    parser.add_argument('--window', type=int, help='Override window (trading days)')
    parser.add_argument('--step', type=int, help='Override step (trading days)')
    parser.add_argument('--panel', type=str, default='data/panels/master_panel.csv', help='Panel path')
    parser.add_argument('--output', '-o', type=str, default='output/temporal', help='Output directory')
    parser.add_argument('--list-lenses', action='store_true', help='List lenses and exit')
    
    args = parser.parse_args()
    
    if args.list_lenses:
        print("\nAvailable Lenses (14 total):")
        print("-" * 40)
        for lens in ALL_LENSES:
            markers = []
            if lens in FAST_LENSES: markers.append("⚡fast")
            if lens in CORE_LENSES: markers.append("★core")
            if lens in HEAVY_LENSES: markers.append("🔥heavy")
            print(f"  {lens:20s} {' '.join(markers)}")
        return
    
    # Load panel
    print(f"Loading: {args.panel}")
    try:
        panel = pd.read_csv(args.panel, parse_dates=['date'], index_col='date')
    except FileNotFoundError:
        print(f"Error: {args.panel} not found")
        sys.exit(1)
    
    # Determine lenses
    lenses = None
    if args.full:
        lenses = ALL_LENSES
    elif args.fast:
        lenses = FAST_LENSES
    elif args.lenses:
        if args.lenses == ['all']:
            lenses = ALL_LENSES
        elif args.lenses == ['fast']:
            lenses = FAST_LENSES
        elif args.lenses == ['core']:
            lenses = CORE_LENSES
        else:
            lenses = args.lenses
    
    # Run
    results = run_temporal_analysis(
        panel,
        resolution=args.resolution,
        window_days=args.window,
        step_days=args.step,
        lenses=lenses,
        parallel=not args.no_parallel,
        workers=args.workers,
    )
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save rankings
    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])
    rank_df.to_csv(output_path / 'rankings.csv')
    
    # Save summary
    summary = get_summary(results)
    summary.to_csv(output_path / 'summary.csv')
    
    print(f"\nResults saved to: {output_path}")
    print("\nTop 10:")
    print(summary.head(10).to_string())


if __name__ == "__main__":
    main()
