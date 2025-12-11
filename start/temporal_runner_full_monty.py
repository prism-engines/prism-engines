"""
Temporal Analysis Runner - FULL MONTY EDITION
==============================================

Aggressive parallel processing optimized for Mac Mini 16GB.
Uses all available cores, chunked memory management, and 
concurrent lens execution to minimize wall-clock time.

This version prioritizes SPEED over conservative resource usage.

Usage:
    # Full send - all lenses, all cores
    python start/temporal_runner.py --full-monty
    
    # Or in Python
    from start.temporal_runner import full_monty
    results = full_monty(panel)
"""

import sys
import os
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import warnings
import gc
import time

warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'engine_core' / 'orchestration'))
sys.path.insert(0, str(PROJECT_ROOT / 'visualization' / 'plotters'))

import pandas as pd
import numpy as np

# Try to get actual CPU count
try:
    # On macOS, this gives performance cores
    CPU_COUNT = os.cpu_count() or 8
except:
    CPU_COUNT = 8


# =============================================================================
# HARDWARE CONFIGURATION - AGGRESSIVE
# =============================================================================

HARDWARE = {
    'name': 'Mac Mini 16GB - FULL MONTY',
    'ram_gb': 16,
    'cpu_cores': CPU_COUNT,
    
    # Aggressive settings
    'max_workers': CPU_COUNT,           # Use ALL cores
    'chunk_size': 20,                   # Windows per chunk
    'memory_limit_gb': 14,              # Leave 2GB for system
    'use_threading_for_io': True,       # Thread pool for I/O bound ops
    'use_multiprocessing_for_compute': True,  # Process pool for CPU bound
    'aggressive_gc': True,              # Force garbage collection between chunks
}

# =============================================================================
# RESOLUTION PRESETS
# =============================================================================

RESOLUTION_PRESETS = {
    'weekly': {
        'frequency': 'W-FRI',
        'window_days': 252,
        'step_days': 5,
        'description': 'Tactical - 1 week steps',
    },
    'monthly': {
        'frequency': 'M',
        'window_days': 252,
        'step_days': 21,
        'description': 'Strategic - 1 month steps (DEFAULT)',
    },
    'quarterly': {
        'frequency': 'Q',
        'window_days': 504,
        'step_days': 63,
        'description': 'Structural - 3 month steps',
    },
    'fine': {
        'frequency': 'W-FRI',
        'window_days': 126,
        'step_days': 5,
        'description': 'High-res - 1 week steps, 6mo window',
    },
    'ultra': {
        'frequency': 'D',
        'window_days': 63,
        'step_days': 1,
        'description': 'Ultra-fine - daily steps (SLOW)',
    },
}

# =============================================================================
# LENS CONFIGURATION
# =============================================================================

ALL_LENSES = [
    'magnitude', 'pca', 'granger', 'dmd', 'influence', 'mutual_info',
    'clustering', 'decomposition', 'wavelet', 'network', 'regime',
    'anomaly', 'transfer_entropy', 'tda',
]

# Categorized by computational cost
FAST_LENSES = ['magnitude', 'pca', 'influence', 'clustering']  # < 0.5s each
MEDIUM_LENSES = ['decomposition', 'regime', 'network', 'dmd', 'wavelet']  # 0.5-2s
HEAVY_LENSES = ['granger', 'mutual_info', 'transfer_entropy', 'tda', 'anomaly']  # 2-10s

CORE_LENSES = FAST_LENSES + ['decomposition', 'regime']


# =============================================================================
# PARALLEL EXECUTION STRATEGIES
# =============================================================================

def _worker_init():
    """Initialize worker process - set up imports once."""
    global _lenses_cache
    _lenses_cache = {}


def _run_single_lens(args: Tuple) -> Dict[str, Any]:
    """Run a single lens on a single window. Minimal overhead."""
    lens_name, window_data_dict, window_idx, window_start, window_end = args
    
    try:
        from engine_core.lenses import get_lens
        
        # Reconstruct DataFrame from dict (for pickling)
        window_data = pd.DataFrame(window_data_dict['data'], 
                                    index=pd.to_datetime(window_data_dict['index']))
        
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
            'success': False,
            'error': str(e),
        }


def _run_lens_batch(args: Tuple) -> List[Dict[str, Any]]:
    """Run all lenses on a single window. Better for memory locality."""
    window_data_dict, window_idx, window_start, window_end, lenses = args
    
    results = []
    try:
        from engine_core.lenses import get_lens
        
        window_data = pd.DataFrame(window_data_dict['data'],
                                    index=pd.to_datetime(window_data_dict['index']))
        
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
                    'success': False,
                    'error': str(e),
                })
    except Exception as e:
        results.append({
            'window_idx': window_idx,
            'success': False,
            'error': f"Window failed: {e}",
        })
    
    return results


def _run_window_chunk(args: Tuple) -> List[Dict[str, Any]]:
    """Process a chunk of windows. Reduces process spawn overhead."""
    chunk_data, lenses = args
    all_results = []
    
    for window_data_dict, window_idx, window_start, window_end in chunk_data:
        batch_args = (window_data_dict, window_idx, window_start, window_end, lenses)
        all_results.extend(_run_lens_batch(batch_args))
    
    return all_results


# =============================================================================
# MAIN ENGINE
# =============================================================================

class FullMontyEngine:
    """
    Maximum performance temporal analysis engine.
    Uses aggressive parallelization and memory management.
    """
    
    def __init__(
        self,
        panel: pd.DataFrame,
        lenses: List[str] = None,
        workers: int = None,
        strategy: str = 'chunked'  # 'chunked', 'per_window', 'per_lens'
    ):
        self.panel = self._prepare_panel(panel)
        self.lenses = lenses or ALL_LENSES
        self.workers = workers or HARDWARE['max_workers']
        self.strategy = strategy
        
        # Stats
        self.stats = {
            'start_time': None,
            'end_time': None,
            'windows_processed': 0,
            'lens_calls': 0,
            'failures': 0,
        }
    
    def _prepare_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Ensure panel is properly formatted."""
        if not isinstance(panel.index, pd.DatetimeIndex):
            if 'date' in panel.columns:
                panel = panel.set_index('date')
                panel.index = pd.to_datetime(panel.index)
            else:
                raise ValueError("Panel must have DatetimeIndex or 'date' column")
        return panel.sort_index()
    
    def _serialize_window(self, window_data: pd.DataFrame) -> dict:
        """Serialize DataFrame for multiprocessing (pickle-friendly)."""
        return {
            'data': window_data.to_dict('list'),
            'index': window_data.index.astype(str).tolist(),
        }
    
    def generate_windows(
        self,
        window_days: int,
        step_days: int
    ) -> List[Tuple[dict, int, str, str]]:
        """Generate all analysis windows as serialized dicts."""
        windows = []
        start_idx = 0
        window_idx = 0
        
        while start_idx + window_days <= len(self.panel):
            end_idx = start_idx + window_days
            window_data = self.panel.iloc[start_idx:end_idx]
            
            windows.append((
                self._serialize_window(window_data),
                window_idx,
                self.panel.index[start_idx].strftime('%Y-%m-%d'),
                self.panel.index[end_idx - 1].strftime('%Y-%m-%d'),
            ))
            
            start_idx += step_days
            window_idx += 1
        
        return windows
    
    def run(
        self,
        window_days: int = 252,
        step_days: int = 21,
        progress_callback: Callable = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run full temporal analysis with maximum parallelization.
        """
        self.stats['start_time'] = time.time()
        
        # Generate windows
        windows = self.generate_windows(window_days, step_days)
        n_windows = len(windows)
        n_lenses = len(self.lenses)
        total_ops = n_windows * n_lenses
        
        if verbose:
            print(f"Windows: {n_windows} | Lenses: {n_lenses} | Total ops: {total_ops}")
            print(f"Workers: {self.workers} | Strategy: {self.strategy}")
        
        # Choose execution strategy
        if self.strategy == 'chunked':
            results = self._run_chunked(windows, progress_callback, verbose)
        elif self.strategy == 'per_window':
            results = self._run_per_window(windows, progress_callback, verbose)
        elif self.strategy == 'per_lens':
            results = self._run_per_lens(windows, progress_callback, verbose)
        else:
            results = self._run_chunked(windows, progress_callback, verbose)
        
        self.stats['end_time'] = time.time()
        elapsed = self.stats['end_time'] - self.stats['start_time']
        
        if verbose:
            print(f"\nCompleted in {elapsed:.1f}s ({total_ops / elapsed:.1f} ops/sec)")
        
        # Organize results
        return self._organize_results(results, windows)
    
    def _run_chunked(
        self,
        windows: List,
        progress_callback: Callable,
        verbose: bool
    ) -> List[Dict]:
        """
        Chunked strategy: Group windows into chunks, process chunks in parallel.
        Best for: Many windows, moderate number of lenses.
        """
        chunk_size = HARDWARE['chunk_size']
        n_windows = len(windows)
        
        # Create chunks
        chunks = []
        for i in range(0, n_windows, chunk_size):
            chunk = windows[i:i + chunk_size]
            chunks.append((chunk, self.lenses))
        
        all_results = []
        completed_chunks = 0
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_run_window_chunk, chunk): idx 
                      for idx, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                
                completed_chunks += 1
                completed_windows = min(completed_chunks * chunk_size, n_windows)
                
                if progress_callback:
                    progress_callback(completed_windows, n_windows)
                
                if HARDWARE['aggressive_gc']:
                    gc.collect()
        
        return all_results
    
    def _run_per_window(
        self,
        windows: List,
        progress_callback: Callable,
        verbose: bool
    ) -> List[Dict]:
        """
        Per-window strategy: Each worker processes one window (all lenses).
        Best for: Moderate windows, many lenses, large data per window.
        """
        work_items = [(w[0], w[1], w[2], w[3], self.lenses) for w in windows]
        
        all_results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_run_lens_batch, item): idx 
                      for idx, item in enumerate(work_items)}
            
            for future in as_completed(futures):
                window_results = future.result()
                all_results.extend(window_results)
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(windows))
        
        return all_results
    
    def _run_per_lens(
        self,
        windows: List,
        progress_callback: Callable,
        verbose: bool
    ) -> List[Dict]:
        """
        Per-lens strategy: Parallelize across (window, lens) pairs.
        Best for: Few windows, heavy lenses, maximum parallelism.
        """
        # Create all (lens, window) pairs
        work_items = [
            (lens, w[0], w[1], w[2], w[3])
            for w in windows
            for lens in self.lenses
        ]
        
        all_results = []
        completed = 0
        total = len(work_items)
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_run_single_lens, item): idx 
                      for idx, item in enumerate(work_items)}
            
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                
                completed += 1
                if progress_callback and completed % 50 == 0:
                    progress_callback(completed, total)
        
        return all_results
    
    def _organize_results(
        self,
        all_results: List[Dict],
        windows: List
    ) -> Dict[str, Any]:
        """Organize raw results into structured output."""
        
        timestamps = [pd.Timestamp(w[3]) for w in windows]
        indicators = list(self.panel.columns)
        n_windows = len(windows)
        
        # Initialize storage
        scores_by_lens = {lens: {ind: [np.nan] * n_windows for ind in indicators} 
                         for lens in self.lenses}
        
        # Process results
        success_count = 0
        for r in all_results:
            if r.get('success', False):
                success_count += 1
                lens = r['lens']
                widx = r['window_idx']
                result = r.get('result', {})
                
                if 'scores' in result:
                    for ind, score in result['scores'].items():
                        if ind in scores_by_lens[lens]:
                            scores_by_lens[lens][ind][widx] = score
        
        self.stats['lens_calls'] = len(all_results)
        self.stats['failures'] = len(all_results) - success_count
        
        # Compute rankings
        rankings_by_lens = {lens: {} for lens in self.lenses}
        
        for lens in self.lenses:
            for widx in range(n_windows):
                window_scores = {
                    ind: scores_by_lens[lens][ind][widx]
                    for ind in indicators
                    if not np.isnan(scores_by_lens[lens][ind][widx])
                }
                
                if window_scores:
                    sorted_inds = sorted(window_scores.keys(), 
                                        key=lambda x: window_scores[x], reverse=True)
                    for rank, ind in enumerate(sorted_inds, 1):
                        if ind not in rankings_by_lens[lens]:
                            rankings_by_lens[lens][ind] = [np.nan] * n_windows
                        rankings_by_lens[lens][ind][widx] = rank
        
        # Consensus rankings
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
        
        elapsed = self.stats['end_time'] - self.stats['start_time']
        
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
                'total_ops': len(all_results),
                'success_count': success_count,
                'success_rate': success_count / len(all_results) if all_results else 0,
                'elapsed_seconds': elapsed,
                'ops_per_second': len(all_results) / elapsed if elapsed > 0 else 0,
                'workers': self.workers,
                'strategy': self.strategy,
            }
        }


# =============================================================================
# API FUNCTIONS
# =============================================================================

def full_monty(
    panel: pd.DataFrame,
    resolution: str = 'monthly',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    THE FULL MONTY - All lenses, all cores, maximum speed.
    
    Args:
        panel: Your data panel
        resolution: 'weekly', 'monthly', 'quarterly', 'fine', 'ultra'
        verbose: Show progress
    
    Returns:
        Complete temporal analysis results
    """
    preset = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS['monthly'])
    
    if verbose:
        print("=" * 60)
        print("🚀 FULL MONTY MODE - ALL LENSES, ALL CORES")
        print("=" * 60)
        print(f"Hardware:    {HARDWARE['name']}")
        print(f"CPU Cores:   {CPU_COUNT}")
        print(f"Resolution:  {resolution} ({preset['description']})")
        print(f"Window:      {preset['window_days']} days")
        print(f"Step:        {preset['step_days']} days")
        print(f"Lenses:      {len(ALL_LENSES)} (ALL)")
        print(f"Indicators:  {len(panel.columns)}")
        print(f"Data points: {len(panel)}")
        print("-" * 60)
    
    engine = FullMontyEngine(
        panel=panel,
        lenses=ALL_LENSES,
        workers=HARDWARE['max_workers'],
        strategy='chunked'
    )
    
    def progress(current, total):
        if verbose:
            pct = current / total * 100
            bar_len = 40
            filled = int(bar_len * current / total)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r[{bar}] {current}/{total} ({pct:.0f}%)", end='', flush=True)
    
    results = engine.run(
        window_days=preset['window_days'],
        step_days=preset['step_days'],
        progress_callback=progress,
        verbose=verbose
    )
    
    if verbose:
        print()
        print("-" * 60)
        meta = results['metadata']
        print(f"✓ Completed: {meta['total_ops']} operations in {meta['elapsed_seconds']:.1f}s")
        print(f"✓ Throughput: {meta['ops_per_second']:.1f} ops/sec")
        print(f"✓ Success rate: {meta['success_rate']:.1%}")
        print("=" * 60)
    
    return results


def run_temporal_analysis(
    panel: pd.DataFrame,
    resolution: str = 'monthly',
    lenses: List[str] = None,
    workers: int = None,
    strategy: str = 'chunked',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Flexible temporal analysis with customizable settings.
    """
    preset = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS['monthly'])
    lenses = lenses or CORE_LENSES
    workers = workers or HARDWARE['max_workers']
    
    if verbose:
        print(f"Resolution: {resolution} | Lenses: {len(lenses)} | Workers: {workers}")
    
    engine = FullMontyEngine(
        panel=panel,
        lenses=lenses,
        workers=workers,
        strategy=strategy
    )
    
    def progress(current, total):
        if verbose:
            pct = current / total * 100
            print(f"\r  Progress: {current}/{total} ({pct:.0f}%)", end='', flush=True)
    
    results = engine.run(
        window_days=preset['window_days'],
        step_days=preset['step_days'],
        progress_callback=progress,
        verbose=False
    )
    
    if verbose:
        print()
        print(f"✓ {results['metadata']['elapsed_seconds']:.1f}s | {results['metadata']['ops_per_second']:.1f} ops/sec")
    
    return results


def run_fast(panel: pd.DataFrame, resolution: str = 'monthly', **kwargs) -> Dict[str, Any]:
    """Fast lenses only - quick results."""
    return run_temporal_analysis(panel, resolution, lenses=FAST_LENSES, **kwargs)


def run_heavy(panel: pd.DataFrame, resolution: str = 'monthly', **kwargs) -> Dict[str, Any]:
    """Heavy lenses only - deep analysis."""
    return run_temporal_analysis(panel, resolution, lenses=HEAVY_LENSES, **kwargs)


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def get_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """Get summary DataFrame."""
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


def quick_start(panel: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """One-liner full analysis."""
    results = full_monty(panel)
    summary = get_summary(results)
    
    print("\nTOP 10:")
    print(summary.head(10).to_string())
    
    return results, summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Temporal Analysis - FULL MONTY EDITION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 FULL MONTY MODE: All 14 lenses, all CPU cores, maximum speed.

Examples:
  python start/temporal_runner.py --full-monty
  python start/temporal_runner.py --resolution weekly --full-monty
  python start/temporal_runner.py --lenses magnitude pca regime --workers 4
  python start/temporal_runner.py --strategy per_lens  # Max parallelism
        """
    )
    
    parser.add_argument('--full-monty', action='store_true', 
                        help='ALL lenses, ALL cores, MAXIMUM speed')
    parser.add_argument('--resolution', '-r', 
                        choices=['weekly', 'monthly', 'quarterly', 'fine', 'ultra'],
                        default='monthly')
    parser.add_argument('--lenses', '-l', nargs='+', 
                        help='Specific lenses (or: all, fast, heavy, core)')
    parser.add_argument('--workers', '-w', type=int, default=CPU_COUNT)
    parser.add_argument('--strategy', '-s', 
                        choices=['chunked', 'per_window', 'per_lens'],
                        default='chunked',
                        help='Parallelization strategy')
    parser.add_argument('--panel', type=str, default='data/panels/master_panel.csv')
    parser.add_argument('--output', '-o', type=str, default='output/temporal')
    parser.add_argument('--benchmark', action='store_true', 
                        help='Run benchmark comparison of strategies')
    
    args = parser.parse_args()
    
    # Load panel
    print(f"Loading: {args.panel}")
    panel = pd.read_csv(args.panel, parse_dates=['date'], index_col='date')
    print(f"Shape: {panel.shape}")
    
    if args.benchmark:
        # Run benchmark of different strategies
        print("\n" + "=" * 60)
        print("STRATEGY BENCHMARK")
        print("=" * 60)
        
        for strategy in ['chunked', 'per_window', 'per_lens']:
            print(f"\n--- {strategy.upper()} ---")
            engine = FullMontyEngine(panel, lenses=CORE_LENSES, strategy=strategy)
            preset = RESOLUTION_PRESETS[args.resolution]
            
            start = time.time()
            results = engine.run(preset['window_days'], preset['step_days'], verbose=False)
            elapsed = time.time() - start
            
            print(f"  Time: {elapsed:.1f}s | {results['metadata']['ops_per_second']:.1f} ops/sec")
        
        return
    
    # Handle lens selection
    lenses = None
    if args.lenses:
        if args.lenses == ['all']:
            lenses = ALL_LENSES
        elif args.lenses == ['fast']:
            lenses = FAST_LENSES
        elif args.lenses == ['heavy']:
            lenses = HEAVY_LENSES
        elif args.lenses == ['core']:
            lenses = CORE_LENSES
        else:
            lenses = args.lenses
    
    # Run analysis
    if args.full_monty:
        results = full_monty(panel, resolution=args.resolution)
    else:
        results = run_temporal_analysis(
            panel,
            resolution=args.resolution,
            lenses=lenses,
            workers=args.workers,
            strategy=args.strategy
        )
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rank_df = pd.DataFrame(results['rankings'], index=results['timestamps'])
    rank_df.to_csv(output_path / 'rankings.csv')
    
    summary = get_summary(results)
    summary.to_csv(output_path / 'summary.csv')
    
    print(f"\nSaved to: {output_path}")
    print("\nTOP 10:")
    print(summary.head(10).to_string())


if __name__ == '__main__':
    main()
