#!/usr/bin/env python3
"""
PRISM Unified Benchmark Runner
==============================
Runs ALL benchmark tests (v1 + v2) from one command.

Usage:
    python benchmark.py           # Run all benchmarks
    python benchmark.py --regen   # Regenerate test data first
    python benchmark.py --v1      # Run only v1 (6 lenses)
    python benchmark.py --v2      # Run only v2 (4 lenses)

Benchmarks:
    v1: Granger, Regime, Clustering, Wavelet, Anomaly, Pure Noise
    v2: PCA, Network, Transfer Entropy, Mutual Info
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'start' else SCRIPT_DIR
BENCHMARK_DIR = PROJECT_ROOT / 'data' / 'benchmark'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Config loader
# -----------------------------------------------------------------------------

def load_config():
    """Load configuration from prism_config.yaml."""
    import yaml
    
    config_paths = [
        SCRIPT_DIR / 'prism_config.yaml',
        PROJECT_ROOT / 'prism_config.yaml',
        PROJECT_ROOT / 'start' / 'prism_config.yaml',
    ]
    
    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
    
    # Defaults if no config found
    return {
        'benchmark': {
            'output_dir': 'data/benchmark',
            'regenerate_data': False,
            'verbose': True,
        }
    }


# -----------------------------------------------------------------------------
# Benchmark Runners
# -----------------------------------------------------------------------------

def run_v1_benchmarks(regenerate: bool = False, verbose: bool = True):
    """Run v1 benchmarks (6 lenses)."""
    import os
    os.chdir(BENCHMARK_DIR)
    
    # Add benchmark dir to path for imports
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    
    # Generate data if needed
    csv_exists = (BENCHMARK_DIR / 'benchmark_clear_leader.csv').exists()
    
    if regenerate or not csv_exists:
        print("\n📊 Generating v1 benchmark data...")
        try:
            from benchmark_generator_v1 import BenchmarkGenerator
            BenchmarkGenerator().generate_all()
        except ImportError:
            # Try without version suffix
            from benchmark_generator import BenchmarkGenerator
            BenchmarkGenerator().generate_all()
    
    # Run benchmarks
    print("\n" + "=" * 60)
    print("BENCHMARK v1: Core Lenses (6)")
    print("=" * 60)
    
    try:
        from benchmark_runner_v1 import BenchmarkRunner
        runner = BenchmarkRunner(str(BENCHMARK_DIR))
        runner.run_all()
        return runner.results
    except ImportError:
        # Try without version suffix
        from benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(str(BENCHMARK_DIR))
        runner.run_all()
        return runner.results


def run_v2_benchmarks(regenerate: bool = False, verbose: bool = True):
    """Run v2 benchmarks (4 lenses)."""
    import os
    os.chdir(BENCHMARK_DIR)
    
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    
    # Generate data if needed
    csv_exists = (BENCHMARK_DIR / 'benchmark_pca.csv').exists()
    
    if regenerate or not csv_exists:
        print("\n📊 Generating v2 benchmark data...")
        from benchmark_generator_v2 import BenchmarkGeneratorV2
        BenchmarkGeneratorV2().generate_all()
    
    # Run benchmarks
    print("\n" + "=" * 60)
    print("BENCHMARK v2: Advanced Lenses (4)")
    print("=" * 60)
    
    from benchmark_runner_v2 import BenchmarkRunnerV2
    runner = BenchmarkRunnerV2(str(BENCHMARK_DIR))
    results = runner.run_all()
    
    return {r.benchmark_name: r.passed for r in results}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PRISM Unified Benchmark Runner')
    parser.add_argument('--regen', action='store_true', help='Regenerate test data')
    parser.add_argument('--v1', action='store_true', help='Run only v1 benchmarks')
    parser.add_argument('--v2', action='store_true', help='Run only v2 benchmarks')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    benchmark_cfg = config.get('benchmark', {})
    
    regenerate = args.regen or benchmark_cfg.get('regenerate_data', False)
    verbose = not args.quiet and benchmark_cfg.get('verbose', True)
    
    # Determine what to run
    run_v1 = args.v1 or (not args.v1 and not args.v2)
    run_v2 = args.v2 or (not args.v1 and not args.v2)
    
    print("=" * 60)
    print("🔬 PRISM UNIFIED BENCHMARK")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    all_results = {}
    
    # Run v1
    if run_v1:
        try:
            results = run_v1_benchmarks(regenerate, verbose)
            all_results.update(results)
        except Exception as e:
            print(f"\n❌ v1 benchmarks failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Run v2
    if run_v2:
        try:
            results = run_v2_benchmarks(regenerate, verbose)
            all_results.update(results)
        except Exception as e:
            print(f"\n❌ v2 benchmarks failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in all_results.values() if v is True)
    failed = sum(1 for v in all_results.values() if v is False)
    skipped = sum(1 for v in all_results.values() if v is None)
    total = len(all_results)
    
    for name, result in all_results.items():
        if result is True:
            print(f"  ✓ {name}")
        elif result is False:
            print(f"  ✗ {name}")
        else:
            print(f"  ⊘ {name}")
    
    print(f"\n  Passed: {passed}/{total}")
    if failed > 0:
        print(f"  Failed: {failed}")
    if skipped > 0:
        print(f"  Skipped: {skipped}")
    
    if passed == total and total > 0:
        print("\n🎉 ALL BENCHMARKS PASSED!")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
