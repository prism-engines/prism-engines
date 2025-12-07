#!/usr/bin/env python3
"""
PRISM Runtime Costs Checker
============================

Purpose: Measure runtime of key engine segments.

This script benchmarks:
- Panel load time
- Coherence engine time (small subset)
- PCA time
- Lens suite time
- HiddenVariationDetector time

Output:
    RUNTIME SUMMARY
    Panel Load: 0.13s
    Coherence: 0.88s
    PCA: 0.04s
    Lenses: 1.22s
    Hidden Variation: 0.19s

Usage:
    python start/performance/check_runtime_costs.py

Notes:
    - Read-only: never writes to the database
    - Exits gracefully if indicators are missing
"""

import sys
import logging
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Callable
from functools import wraps

import pandas as pd
import numpy as np

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration
NUM_BENCHMARK_INDICATORS = 10
COHERENCE_WINDOW = 63
PCA_COMPONENTS = 3


def benchmark(func: Callable) -> Callable:
    """
    Decorator to benchmark a function's execution time.

    Args:
        func: Function to benchmark

    Returns:
        Wrapped function that returns (result, timing_info)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before timing
        gc.collect()

        start_time = time.perf_counter()
        start_memory = get_memory_usage()

        try:
            result = func(*args, **kwargs)
            error = None
        except Exception as e:
            result = None
            error = str(e)

        end_time = time.perf_counter()
        end_memory = get_memory_usage()

        timing = {
            "duration_s": end_time - start_time,
            "duration_ms": (end_time - start_time) * 1000,
            "memory_delta_mb": end_memory - start_memory,
            "error": error
        }

        return result, timing

    return wrapper


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


@benchmark
def benchmark_panel_load() -> pd.DataFrame:
    """Benchmark panel loading time."""
    from data.sql.db_connector import load_all_indicators_wide

    df = load_all_indicators_wide()

    # Select subset
    if not df.empty and len(df.columns) > NUM_BENCHMARK_INDICATORS:
        coverage = (~df.isna()).mean()
        best_cols = coverage.nlargest(NUM_BENCHMARK_INDICATORS).index.tolist()
        df = df[best_cols]

    return df


@benchmark
def benchmark_coherence(df: pd.DataFrame) -> pd.Series:
    """Benchmark coherence computation."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    # Compute returns
    returns_df = df.pct_change().dropna()

    if len(returns_df) < COHERENCE_WINDOW:
        return pd.Series(dtype=float)

    # Rolling coherence
    coherence_values = []
    dates = []

    for i in range(COHERENCE_WINDOW, len(returns_df)):
        window_data = returns_df.iloc[i-COHERENCE_WINDOW:i]
        corr_matrix = window_data.corr()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_corrs = corr_matrix.values[mask]
        coherence = np.nanmean(np.abs(upper_corrs))

        coherence_values.append(coherence)
        dates.append(returns_df.index[i])

    return pd.Series(coherence_values, index=dates)


@benchmark
def benchmark_pca(df: pd.DataFrame) -> Dict[str, Any]:
    """Benchmark PCA computation."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    if df is None or df.empty:
        return {}

    df_clean = df.dropna()

    if len(df_clean) < 50:
        return {}

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean)

    n_comp = min(PCA_COMPONENTS, len(df_clean.columns))
    pca = PCA(n_components=n_comp)
    pca.fit(data_scaled)

    return {
        "components": pca.components_,
        "explained_variance": pca.explained_variance_ratio_
    }


@benchmark
def benchmark_lens_suite(df: pd.DataFrame) -> Dict[str, Any]:
    """Benchmark basic lens suite."""
    results = {}

    if df is None or df.empty:
        return results

    # Prepare data for lenses (add 'date' column)
    df_lens = df.reset_index()
    if 'index' in df_lens.columns:
        df_lens = df_lens.rename(columns={'index': 'date'})

    # Test basic lenses only
    basic_lenses = ['magnitude', 'pca', 'clustering']

    for lens_name in basic_lenses:
        try:
            if lens_name == 'magnitude':
                from engine_core.lenses.magnitude_lens import MagnitudeLens
                lens = MagnitudeLens()
            elif lens_name == 'pca':
                from engine_core.lenses.pca_lens import PCALens
                lens = PCALens()
            elif lens_name == 'clustering':
                from engine_core.lenses.clustering_lens import ClusteringLens
                lens = ClusteringLens()
            else:
                continue

            lens_result = lens.analyze(df_lens)
            results[lens_name] = {"status": "ok", "keys": list(lens_result.keys())}

        except Exception as e:
            results[lens_name] = {"status": "error", "error": str(e)}

    return results


@benchmark
def benchmark_hvd(df: pd.DataFrame) -> Dict[str, Any]:
    """Benchmark HiddenVariationDetector."""
    from analysis.hidden_variation_detector import build_hvd_report
    from data.family_manager import FamilyManager

    if df is None or df.empty:
        return {}

    try:
        fm = FamilyManager()
        report = build_hvd_report(
            panel=df,
            family_manager=fm,
            similarity_threshold=0.90,
            cluster_method='auto',
            n_clusters=4,
            min_overlap=252
        )

        return {
            "clusters": len(report.clusters),
            "high_similarity_pairs": len(report.high_similarity_pairs),
            "warnings": len(report.warnings)
        }

    except Exception as e:
        return {"error": str(e)}


def format_duration(seconds: float) -> str:
    """Format duration for display."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f}us"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def print_report(benchmarks: Dict[str, Dict[str, Any]]) -> None:
    """Print the runtime costs report."""

    print("\n" + "=" * 60)
    print("RUNTIME SUMMARY")
    print("=" * 60)
    print(f"Report time: {datetime.now().isoformat()}")

    # Calculate total
    total_time = sum(
        b.get("duration_s", 0)
        for b in benchmarks.values()
        if "duration_s" in b
    )

    # Print table
    print("\n{:<25} {:>15} {:>12} {:>10}".format(
        "Operation", "Duration", "Memory", "Status"
    ))
    print("-" * 62)

    for name, timing in benchmarks.items():
        duration = timing.get("duration_s", 0)
        duration_str = format_duration(duration)

        memory = timing.get("memory_delta_mb", 0)
        memory_str = f"{memory:+.1f}MB" if memory != 0 else "N/A"

        if timing.get("error"):
            status = "ERROR"
        else:
            status = "OK"

        # Calculate percentage of total
        pct = (duration / total_time * 100) if total_time > 0 else 0

        print("{:<25} {:>15} {:>12} {:>10}".format(
            name, f"{duration_str} ({pct:.0f}%)", memory_str, status
        ))

    print("-" * 62)
    print("{:<25} {:>15}".format("TOTAL", format_duration(total_time)))

    # Print errors if any
    errors = [(name, t["error"]) for name, t in benchmarks.items() if t.get("error")]
    if errors:
        print("\n--- Errors ---")
        for name, error in errors:
            print(f"  {name}: {error[:60]}")

    # Print breakdown bar chart (ASCII)
    print("\n--- Time Breakdown ---")
    max_bar_len = 40

    for name, timing in sorted(
        benchmarks.items(),
        key=lambda x: x[1].get("duration_s", 0),
        reverse=True
    ):
        duration = timing.get("duration_s", 0)
        if total_time > 0:
            bar_len = int((duration / total_time) * max_bar_len)
            bar = "#" * bar_len
            pct = duration / total_time * 100
            print(f"  {name:<20} [{bar:<{max_bar_len}}] {pct:>5.1f}%")

    print("\n" + "=" * 60)


def main() -> int:
    """
    Main entry point for runtime costs check.

    Returns:
        0 on success, 1 on error
    """
    logger.info("Starting runtime costs benchmark...")

    # Import db_path lazily
    from data.sql.db_connector import get_db_path
    logger.info(f"Database path: {get_db_path()}")

    benchmarks = {}

    try:
        # Benchmark panel load
        logger.info("Benchmarking panel load...")
        df, timing = benchmark_panel_load()
        benchmarks["Panel Load"] = timing

        if df is None or df.empty:
            print("\n[!] WARNING: No data loaded. Cannot continue benchmarks.")
            return 1

        logger.info(f"Panel shape: {df.shape}")

        # Benchmark coherence
        logger.info("Benchmarking coherence engine...")
        _, timing = benchmark_coherence(df)
        benchmarks["Coherence Engine"] = timing

        # Benchmark PCA
        logger.info("Benchmarking PCA...")
        _, timing = benchmark_pca(df)
        benchmarks["PCA"] = timing

        # Benchmark lens suite
        logger.info("Benchmarking lens suite...")
        _, timing = benchmark_lens_suite(df)
        benchmarks["Lens Suite (basic)"] = timing

        # Benchmark HVD
        logger.info("Benchmarking HVD...")
        _, timing = benchmark_hvd(df)
        benchmarks["Hidden Variation"] = timing

        # Print report
        print_report(benchmarks)

        # Check for errors
        errors = [name for name, t in benchmarks.items() if t.get("error")]
        if errors:
            logger.warning(f"Benchmarks completed with {len(errors)} error(s)")
            return 1

        logger.info("Runtime costs benchmark completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in runtime costs benchmark: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
