"""
Engine Performance Benchmarks for PRISM Engine

Measures runtime, memory footprint, and throughput for the temporal engine
using real data from the database.

Benchmark profiles:
- standard: Baseline configuration with core lenses
- powerful: Full lens suite with finer stepping
- reduced: Smaller panel for scaling comparison
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import sys
import traceback
import gc

import numpy as np
import pandas as pd


@dataclass
class PerformanceResult:
    """Result from a single performance benchmark."""

    name: str
    profile: str
    n_indicators: int
    n_lenses: int
    n_windows: int
    runtime_seconds: float
    result_size_mb: float = 0.0
    panel_size_mb: float = 0.0
    notes: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    lenses_used: list[str] = field(default_factory=list)
    date_range: tuple[str, str] | None = None


# Benchmark configurations (profiles)
BENCHMARK_PROFILES = {
    "standard": {
        "description": "Baseline full run with core lenses",
        "lenses": ["magnitude", "pca", "influence", "clustering"],
        "window_days": 252,  # 1 year
        "step_days": 21,  # ~monthly
        "max_indicators": None,  # All indicators
    },
    "powerful": {
        "description": "Full lens suite with finer stepping",
        "lenses": ["magnitude", "pca", "influence", "clustering", "network"],
        "window_days": 252,
        "step_days": 10,  # Finer stepping
        "max_indicators": None,
    },
    "reduced": {
        "description": "Reduced panel for scaling comparison",
        "lenses": ["magnitude", "pca", "influence"],
        "window_days": 252,
        "step_days": 21,
        "max_indicators": 25,  # Only first 25 indicators
    },
    "minimal": {
        "description": "Minimal config for quick testing",
        "lenses": ["magnitude", "pca"],
        "window_days": 126,  # 6 months
        "step_days": 42,  # ~bi-monthly
        "max_indicators": 15,
    },
}


def _estimate_size_mb(obj) -> float:
    """Estimate memory size of an object in MB."""
    try:
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum() / (1024 * 1024)
        elif isinstance(obj, dict):
            # Rough estimate for nested dicts
            import json
            try:
                return len(json.dumps(obj, default=str)) / (1024 * 1024)
            except (TypeError, ValueError):
                return sys.getsizeof(obj) / (1024 * 1024)
        else:
            return sys.getsizeof(obj) / (1024 * 1024)
    except Exception:
        return 0.0


def _load_panel_data(
    max_indicators: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load panel data from the database.

    Returns
    -------
    tuple
        (panel_dataframe, db_path)
    """
    from data.sql.db_path import get_db_path
    from data.sql.db_connector import load_all_indicators_wide

    db_path = get_db_path()
    panel = load_all_indicators_wide(start_date=start_date, end_date=end_date)

    if max_indicators is not None and len(panel.columns) > max_indicators:
        # Take first N indicators (deterministic for reproducibility)
        panel = panel.iloc[:, :max_indicators]

    return panel, db_path


def run_single_performance_benchmark(
    profile_name: str,
    profile_config: dict,
    verbose: bool = False,
) -> PerformanceResult:
    """
    Run a single performance benchmark.

    Parameters
    ----------
    profile_name : str
        Name of the benchmark profile.
    profile_config : dict
        Configuration for this profile.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    PerformanceResult
        Structured result with timing and metrics.
    """
    result = PerformanceResult(
        name=profile_name,
        profile=profile_name,
        n_indicators=0,
        n_lenses=len(profile_config.get("lenses", [])),
        n_windows=0,
        runtime_seconds=0.0,
        lenses_used=profile_config.get("lenses", []),
    )

    if verbose:
        print(f"\n  Running profile: {profile_name}")
        print(f"    {profile_config.get('description', '')}")

    try:
        # Load data
        if verbose:
            print("    Loading panel data...", end=" ")

        load_start = time.time()
        panel, db_path = _load_panel_data(
            max_indicators=profile_config.get("max_indicators"),
        )
        load_time = time.time() - load_start

        if verbose:
            print(f"done ({load_time:.2f}s)")

        result.n_indicators = len(panel.columns)
        result.panel_size_mb = _estimate_size_mb(panel)

        if verbose:
            print(f"    Panel: {len(panel)} rows x {result.n_indicators} columns ({result.panel_size_mb:.1f} MB)")

        # Set date range
        if len(panel) > 0:
            result.date_range = (
                str(panel.index[0].date()),
                str(panel.index[-1].date()),
            )

        # Run temporal analysis
        if verbose:
            print("    Running temporal analysis...")

        from engine_core.orchestration.temporal_analysis import TemporalEngine

        gc.collect()  # Clean memory before benchmark

        analysis_start = time.time()

        temporal = TemporalEngine(panel, lenses=profile_config.get("lenses"))
        analysis_result = temporal.run_rolling_analysis(
            window_days=profile_config.get("window_days", 252),
            step_days=profile_config.get("step_days", 21),
        )

        result.runtime_seconds = time.time() - analysis_start

        # Extract metrics
        result.n_windows = analysis_result["metadata"]["n_windows"]
        result.result_size_mb = _estimate_size_mb(analysis_result)

        # Add notes
        result.notes.append(f"DB: {db_path}")
        result.notes.append(f"Load time: {load_time:.2f}s")

        if analysis_result["metadata"].get("date_range"):
            dr = analysis_result["metadata"]["date_range"]
            result.notes.append(f"Analysis range: {dr[0].date()} to {dr[1].date()}")

        if verbose:
            print(f"    Completed in {result.runtime_seconds:.2f}s")
            print(f"    Windows: {result.n_windows}, Result size: {result.result_size_mb:.1f} MB")

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        result.notes.append(f"Error: {str(e)}")
        if verbose:
            print(f"    FAILED: {e}")
            traceback.print_exc()

    return result


def run_engine_performance_benchmarks(
    profiles: list[str] | None = None,
    verbose: bool = False,
) -> list[PerformanceResult]:
    """
    Run all performance benchmarks.

    Parameters
    ----------
    profiles : list[str], optional
        Which profiles to run. Defaults to all.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    list[PerformanceResult]
        Results for all benchmarks.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PRISM ENGINE PERFORMANCE BENCHMARKS")
        print("=" * 60)

    if profiles is None:
        profiles = list(BENCHMARK_PROFILES.keys())

    if verbose:
        print(f"Running {len(profiles)} profiles: {profiles}")

    results = []

    for profile_name in profiles:
        if profile_name not in BENCHMARK_PROFILES:
            if verbose:
                print(f"\n  Unknown profile: {profile_name}, skipping")
            continue

        profile_config = BENCHMARK_PROFILES[profile_name]
        result = run_single_performance_benchmark(
            profile_name=profile_name,
            profile_config=profile_config,
            verbose=verbose,
        )
        results.append(result)

        # Clean up between benchmarks
        gc.collect()

    if verbose:
        # Summary
        print("\n" + "-" * 60)
        print("PERFORMANCE SUMMARY")
        print("-" * 60)

        for r in results:
            status = "PASS" if r.success else "FAIL"
            print(f"  {r.name}: {status}")
            print(f"    Runtime: {r.runtime_seconds:.2f}s")
            print(f"    Indicators: {r.n_indicators}, Windows: {r.n_windows}")
            print(f"    Panel: {r.panel_size_mb:.1f} MB, Result: {r.result_size_mb:.1f} MB")

    return results


def run_scaling_benchmark(
    indicator_counts: list[int] | None = None,
    verbose: bool = False,
) -> list[PerformanceResult]:
    """
    Run scaling analysis to understand performance vs data size.

    Parameters
    ----------
    indicator_counts : list[int], optional
        Number of indicators to test at each level.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    list[PerformanceResult]
        Results showing scaling behavior.
    """
    if indicator_counts is None:
        indicator_counts = [10, 25, 50, 75, 100]

    if verbose:
        print("\n" + "=" * 60)
        print("SCALING BENCHMARK")
        print("=" * 60)

    results = []

    for count in indicator_counts:
        profile_config = {
            "description": f"Scaling test with {count} indicators",
            "lenses": ["magnitude", "pca", "influence"],
            "window_days": 252,
            "step_days": 21,
            "max_indicators": count,
        }

        result = run_single_performance_benchmark(
            profile_name=f"scaling_{count}",
            profile_config=profile_config,
            verbose=verbose,
        )
        results.append(result)
        gc.collect()

    if verbose:
        print("\n" + "-" * 60)
        print("SCALING RESULTS")
        print("-" * 60)
        print(f"{'Indicators':>12} {'Runtime (s)':>12} {'Windows':>10} {'MB/ind':>10}")
        print("-" * 46)

        for r in results:
            mb_per_ind = r.result_size_mb / r.n_indicators if r.n_indicators > 0 else 0
            print(f"{r.n_indicators:>12} {r.runtime_seconds:>12.2f} {r.n_windows:>10} {mb_per_ind:>10.3f}")

    return results


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Engine Performance Benchmarks")
    parser.add_argument("--profile", type=str, help="Specific profile to run")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.scaling:
        results = run_scaling_benchmark(verbose=args.verbose or True)
    elif args.profile:
        results = run_engine_performance_benchmarks(profiles=[args.profile], verbose=args.verbose or True)
    else:
        results = run_engine_performance_benchmarks(verbose=args.verbose or True)

    print(f"\nCompleted {len(results)} benchmark(s)")
