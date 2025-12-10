#!/usr/bin/env python3
"""
PRISM Full Monty Analysis
=========================
The complete analysis suite - run everything, measure everything, compare everything.
Designed for powerful machines with time to spare.

What it runs:
  1. All 14 lenses on full date range
  2. Temporal analysis with multiple windows
  3. Historical period comparisons (2008 crisis, COVID, etc.)
  4. Cross-period stability analysis
  5. Full benchmark validation
  6. Performance profiling

Usage:
    python full_monty.py                 # Run everything
    python full_monty.py --skip-temporal # Skip slow temporal analysis
    python full_monty.py --quick         # Fast mode (fewer windows, recent data only)

Output:
    - All results saved to database
    - Summary report generated
    - Performance metrics logged
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'start' else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Historical periods of interest
HISTORICAL_PERIODS = {
    'dot_com_crash': ('2000-01-01', '2002-12-31'),
    'pre_gfc': ('2005-01-01', '2007-06-30'),
    'gfc_crisis': ('2007-07-01', '2009-03-31'),
    'gfc_recovery': ('2009-04-01', '2012-12-31'),
    'bull_market': ('2013-01-01', '2019-12-31'),
    'covid_crash': ('2020-01-01', '2020-04-30'),
    'covid_recovery': ('2020-05-01', '2021-12-31'),
    'inflation_era': ('2022-01-01', '2023-12-31'),
    'recent': ('2024-01-01', None),  # None = today
}

# Temporal windows (in trading days)
TEMPORAL_WINDOWS = {
    'full': [21, 63, 126, 252, 504],      # 1mo, 3mo, 6mo, 1yr, 2yr
    'quick': [63, 252],                    # 3mo, 1yr only
}

# All lenses to run
ALL_LENSES = [
    'granger', 'regime', 'clustering', 'wavelet', 'anomaly',
    'pca', 'network', 'transfer_entropy', 'mutual_info', 'magnitude',
    'decomposition', 'dmd', 'influence', 'tda'
]


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class Timer:
    """Simple timing context manager."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start = None
        self.elapsed = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        
    def __str__(self):
        if self.elapsed is None:
            return f"{self.name}: running..."
        elif self.elapsed < 60:
            return f"{self.name}: {self.elapsed:.1f}s"
        else:
            mins = int(self.elapsed // 60)
            secs = self.elapsed % 60
            return f"{self.name}: {mins}m {secs:.0f}s"


class PerformanceLog:
    """Track performance across all operations."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.start_time = time.time()
    
    def log(self, name: str, seconds: float):
        self.timings[name] = seconds
    
    def total_elapsed(self) -> float:
        return time.time() - self.start_time
    
    def summary(self) -> str:
        lines = ["\n⏱️  PERFORMANCE SUMMARY", "=" * 50]
        
        # Sort by time descending
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for name, secs in sorted_timings:
            if secs < 60:
                lines.append(f"   {name:<35} {secs:>8.1f}s")
            else:
                mins = int(secs // 60)
                s = secs % 60
                lines.append(f"   {name:<35} {mins:>5}m {s:>4.0f}s")
        
        lines.append("-" * 50)
        total = self.total_elapsed()
        mins = int(total // 60)
        secs = total % 60
        lines.append(f"   {'TOTAL':<35} {mins:>5}m {secs:>4.0f}s")
        
        return "\n".join(lines)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_full_analysis(config: dict, perf: PerformanceLog) -> Dict[str, Any]:
    """Run all lenses on full date range."""
    from analyze import AnalysisRunner, load_data, preprocess_data, normalize_date_column, LENS_REGISTRY
    
    print("\n" + "=" * 60)
    print("📊 FULL ANALYSIS - All Lenses")
    print("=" * 60)
    
    with Timer("Full Analysis") as t:
        # Load and preprocess
        df = load_data(config)
        df = normalize_date_column(df)
        min_cov = config.get('data', {}).get('min_coverage', 0.7)
        df = preprocess_data(df, min_coverage=min_cov)
        
        print(f"   Data: {len(df)} rows × {len(df.columns)-1} indicators")
        
        # Run all lenses
        runner = AnalysisRunner(config)
        runner.run_all(df, ALL_LENSES)
        
        # Save to DB
        try:
            from data.sql.db_connector import save_analysis_run
            
            normalize_methods = {
                name: LENS_REGISTRY.get(name, {}).get('normalize', 'none')
                for name in ALL_LENSES
            }
            
            run_id = save_analysis_run(
                start_date=config.get('data', {}).get('start_date', '2000-01-01'),
                end_date=config.get('data', {}).get('end_date'),
                n_indicators=len([c for c in df.columns if c != 'date']),
                n_rows=len(df),
                n_lenses=len(runner.results),
                n_errors=len(runner.errors),
                config=config,
                runtime_seconds=t.elapsed,
                lens_results=runner.results,
                lens_errors=runner.errors,
                rankings=runner.rankings,
                consensus=runner.get_consensus_rankings(),
                normalize_methods=normalize_methods,
            )
            print(f"\n   💾 Saved: run_id={run_id}")
        except Exception as e:
            print(f"\n   ⚠️  DB save failed: {e}")
            run_id = None
    
    perf.log("Full Analysis", t.elapsed)
    
    return {
        'run_id': run_id,
        'n_lenses': len(runner.results),
        'n_errors': len(runner.errors),
        'consensus': runner.get_consensus_rankings(),
    }


def run_historical_periods(config: dict, perf: PerformanceLog) -> Dict[str, Any]:
    """Run analysis on each historical period."""
    from analyze import AnalysisRunner, load_data, preprocess_data, normalize_date_column, LENS_REGISTRY
    from data.sql.db_connector import save_analysis_run
    
    print("\n" + "=" * 60)
    print("📅 HISTORICAL PERIOD ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    for period_name, (start, end) in HISTORICAL_PERIODS.items():
        print(f"\n   ▶ {period_name} ({start} → {end or 'today'})...")
        
        with Timer(period_name) as t:
            try:
                # Create period-specific config
                period_config = config.copy()
                period_config['data'] = config.get('data', {}).copy()
                period_config['data']['start_date'] = start
                period_config['data']['end_date'] = end
                
                # Load data for this period
                df = load_data(period_config)
                df = normalize_date_column(df)
                min_cov = config.get('data', {}).get('min_coverage', 0.5)  # Lower for shorter periods
                df = preprocess_data(df, min_coverage=min_cov)
                
                if len(df) < 50:
                    print(f"     ⚠️  Insufficient data ({len(df)} rows)")
                    continue
                
                # Run subset of lenses (faster)
                quick_lenses = ['pca', 'clustering', 'regime', 'anomaly', 'network']
                runner = AnalysisRunner(period_config)
                runner.run_all(df, quick_lenses)
                
                # Save to DB
                normalize_methods = {
                    name: LENS_REGISTRY.get(name, {}).get('normalize', 'none')
                    for name in quick_lenses
                }
                
                run_id = save_analysis_run(
                    start_date=start,
                    end_date=end,
                    n_indicators=len([c for c in df.columns if c != 'date']),
                    n_rows=len(df),
                    n_lenses=len(runner.results),
                    n_errors=len(runner.errors),
                    config={'period': period_name, **period_config},
                    runtime_seconds=t.elapsed,
                    lens_results=runner.results,
                    lens_errors=runner.errors,
                    rankings=runner.rankings,
                    consensus=runner.get_consensus_rankings(),
                    normalize_methods=normalize_methods,
                )
                
                results[period_name] = {
                    'run_id': run_id,
                    'rows': len(df),
                    'top_3': list(runner.get_consensus_rankings().head(3).index),
                }
                print(f"     ✓ run_id={run_id}, top: {results[period_name]['top_3']}")
                
            except Exception as e:
                print(f"     ✗ Error: {e}")
                results[period_name] = {'error': str(e)}
        
        perf.log(f"Period: {period_name}", t.elapsed)
    
    return results


def run_temporal_analysis(config: dict, windows: List[int], perf: PerformanceLog) -> Dict[str, Any]:
    """Run rolling temporal analysis."""
    print("\n" + "=" * 60)
    print("🕐 TEMPORAL ANALYSIS")
    print("=" * 60)
    
    # Check if temporal.py exists
    temporal_path = SCRIPT_DIR / 'temporal.py'
    if not temporal_path.exists():
        print("   ⚠️  temporal.py not found, skipping")
        return {'skipped': True}
    
    with Timer("Temporal Analysis") as t:
        try:
            import subprocess
            
            # Build command
            cmd = [
                sys.executable, str(temporal_path),
                '--windows', *[str(w) for w in windows],
            ]
            
            start_date = config.get('data', {}).get('start_date')
            if start_date:
                cmd.extend(['--start', start_date])
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
            
            if result.returncode == 0:
                print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
                status = 'success'
            else:
                print(f"   ✗ Error: {result.stderr[:500]}")
                status = 'error'
                
        except Exception as e:
            print(f"   ✗ Exception: {e}")
            status = 'exception'
    
    perf.log("Temporal Analysis", t.elapsed)
    
    return {'status': status, 'windows': windows}


def run_benchmarks(perf: PerformanceLog) -> Dict[str, Any]:
    """Run benchmark validation suite."""
    print("\n" + "=" * 60)
    print("🔬 BENCHMARK VALIDATION")
    print("=" * 60)
    
    benchmark_path = SCRIPT_DIR / 'benchmark.py'
    if not benchmark_path.exists():
        print("   ⚠️  benchmark.py not found, skipping")
        return {'skipped': True}
    
    with Timer("Benchmarks") as t:
        try:
            import subprocess
            
            result = subprocess.run(
                [sys.executable, str(benchmark_path)],
                capture_output=True, text=True, cwd=str(SCRIPT_DIR)
            )
            
            # Extract pass/fail from output
            output = result.stdout
            if 'ALL BENCHMARKS PASSED' in output or 'ALL PASSED' in output:
                status = 'all_passed'
            elif 'Passed:' in output:
                # Extract score
                import re
                match = re.search(r'Passed:\s*(\d+)/(\d+)', output)
                if match:
                    status = f"{match.group(1)}/{match.group(2)}"
                else:
                    status = 'completed'
            else:
                status = 'unknown'
            
            # Print last part of output
            lines = output.strip().split('\n')
            for line in lines[-15:]:
                print(f"   {line}")
                
        except Exception as e:
            print(f"   ✗ Exception: {e}")
            status = 'exception'
    
    perf.log("Benchmarks", t.elapsed)
    
    return {'status': status}


def compare_periods(period_results: Dict[str, Any]) -> None:
    """Compare top indicators across periods."""
    print("\n" + "=" * 60)
    print("🔄 CROSS-PERIOD COMPARISON")
    print("=" * 60)
    
    # Collect top indicators per period
    period_tops = {}
    for period, result in period_results.items():
        if 'top_3' in result:
            period_tops[period] = result['top_3']
    
    if len(period_tops) < 2:
        print("   Not enough periods to compare")
        return
    
    # Find indicators that appear across periods
    all_indicators = []
    for tops in period_tops.values():
        all_indicators.extend(tops)
    
    from collections import Counter
    indicator_counts = Counter(all_indicators)
    
    print("\n   Indicators appearing in top 3 across periods:")
    print("-" * 40)
    for indicator, count in indicator_counts.most_common(10):
        periods = [p for p, tops in period_tops.items() if indicator in tops]
        print(f"   {indicator:<20} {count}x  ({', '.join(periods[:3])}...)")


def generate_report(
    full_results: Dict,
    period_results: Dict,
    temporal_results: Dict,
    benchmark_results: Dict,
    perf: PerformanceLog
) -> str:
    """Generate summary report."""
    
    lines = [
        "",
        "=" * 70,
        "🎯 PRISM FULL MONTY - SUMMARY REPORT",
        f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "📊 FULL ANALYSIS",
        "-" * 40,
        f"   Run ID: {full_results.get('run_id')}",
        f"   Lenses: {full_results.get('n_lenses')} successful, {full_results.get('n_errors')} errors",
    ]
    
    consensus = full_results.get('consensus')
    if consensus is not None and not consensus.empty:
        lines.append("\n   Top 10 Consensus:")
        for i, (ind, row) in enumerate(consensus.head(10).iterrows(), 1):
            lines.append(f"      {i:>2}. {ind:<20} {row.get('mean_score', 0):.3f}")
    
    lines.extend([
        "",
        "📅 HISTORICAL PERIODS",
        "-" * 40,
    ])
    for period, result in period_results.items():
        if 'error' in result:
            lines.append(f"   {period:<20} ✗ {result['error'][:30]}")
        elif 'top_3' in result:
            lines.append(f"   {period:<20} ✓ Top: {', '.join(result['top_3'])}")
    
    lines.extend([
        "",
        "🔬 BENCHMARKS",
        "-" * 40,
        f"   Status: {benchmark_results.get('status', 'not run')}",
    ])
    
    lines.append(perf.summary())
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PRISM Full Monty Analysis')
    parser.add_argument('--skip-temporal', action='store_true', help='Skip temporal analysis')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip benchmark validation')
    parser.add_argument('--skip-historical', action='store_true', help='Skip historical periods')
    parser.add_argument('--quick', action='store_true', help='Quick mode (recent data, fewer windows)')
    parser.add_argument('--start', type=str, help='Override start date')
    parser.add_argument('--coverage', type=float, default=0.7, help='Min coverage threshold')
    
    args = parser.parse_args()
    
    # Setup
    perf = PerformanceLog()
    
    print("=" * 70)
    print("🎬 PRISM FULL MONTY")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   Mode:", "QUICK" if args.quick else "FULL")
    print("=" * 70)
    
    # Build config
    config = {
        'data': {
            'start_date': args.start or ('2020-01-01' if args.quick else '2000-01-01'),
            'end_date': None,
            'min_coverage': args.coverage,
        },
        'analysis': {
            'save_results': True,
        }
    }
    
    # 1. Full analysis
    full_results = run_full_analysis(config, perf)
    
    # 2. Historical periods
    if args.skip_historical:
        period_results = {'skipped': True}
    else:
        period_results = run_historical_periods(config, perf)
        compare_periods(period_results)
    
    # 3. Temporal analysis
    if args.skip_temporal:
        temporal_results = {'skipped': True}
    else:
        windows = TEMPORAL_WINDOWS['quick'] if args.quick else TEMPORAL_WINDOWS['full']
        temporal_results = run_temporal_analysis(config, windows, perf)
    
    # 4. Benchmarks
    if args.skip_benchmarks:
        benchmark_results = {'skipped': True}
    else:
        benchmark_results = run_benchmarks(perf)
    
    # Generate report
    report = generate_report(full_results, period_results, temporal_results, benchmark_results, perf)
    print(report)
    
    # Save report to file
    report_path = SCRIPT_DIR / 'output' / 'full_monty_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved: {report_path}")
    
    print("\n" + "=" * 70)
    print("🎬 FULL MONTY COMPLETE!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
