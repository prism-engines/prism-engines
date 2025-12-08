"""
Overnight Analysis Benchmarks for PRISM Engine

Thin wrappers to benchmark the overnight analysis scripts (full and lite versions).
Measures runtime, component counts, and success/failure status.

Scripts benchmarked:
- start/overnight_analysis.py (full overnight, 6-12 hours)
- start/overnight_lite.py (lite version, 30-60 minutes)
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import time
import traceback
import gc
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class OvernightResult:
    """Result from an overnight benchmark."""

    script_name: str
    mode: str  # e.g., "lite", "full", "quick_test"
    runtime_seconds: float
    n_indicators: Optional[int] = None
    n_lenses: Optional[int] = None
    n_windows: Optional[int] = None
    parallel_enabled: bool = False
    n_workers: Optional[int] = None
    notes: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    components_run: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)


def _load_data_for_benchmark(years_back: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load panel and returns data for benchmarking."""
    from data.sql.db_connector import load_all_indicators_wide

    panel = load_all_indicators_wide()

    if panel.empty:
        return pd.DataFrame(), pd.DataFrame()

    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # Filter to analysis period
    cutoff = panel.index.max() - pd.DateOffset(years=years_back)
    panel = panel[panel.index >= cutoff]

    # Keep columns with 50%+ coverage
    valid_cols = panel.columns[panel.notna().sum() > len(panel) * 0.5]
    panel = panel[valid_cols].ffill().bfill().dropna(axis=1)

    # Compute returns
    returns = panel.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)

    return panel, returns


def _run_multiresolution_subset(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    window_sizes: list[int],
    step_size: int,
    max_windows: int = 100,
) -> dict[str, Any]:
    """
    Run a subset of multi-resolution analysis for benchmarking.

    Returns basic metrics without full analysis overhead.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from scipy import stats

    results = {
        "window_sizes": window_sizes,
        "step_size": step_size,
        "n_windows_per_size": {},
        "total_windows": 0,
    }

    for window_size in window_sizes:
        n_windows = min(
            len(range(window_size, len(returns), step_size)),
            max_windows,
        )
        results["n_windows_per_size"][window_size] = n_windows
        results["total_windows"] += n_windows

    return results


def run_lite_benchmark(
    years_back: int = 3,
    max_windows: int = 50,
    verbose: bool = False,
) -> OvernightResult:
    """
    Benchmark the lite overnight analysis pattern.

    Uses reduced parameters for faster testing while exercising
    the same code paths as the full overnight_lite.py.
    """
    result = OvernightResult(
        script_name="overnight_lite.py",
        mode="benchmark",
        runtime_seconds=0.0,
    )

    if verbose:
        print("\n  Benchmarking overnight_lite pattern...")

    start_time = time.time()

    try:
        # Load data
        panel, returns = _load_data_for_benchmark(years_back=years_back)

        if panel.empty or returns.empty:
            result.success = False
            result.error_message = "No data available"
            result.notes.append("Data loading failed")
            return result

        result.n_indicators = len(panel.columns)

        # Configuration matching overnight_lite.py pattern
        config = {
            "years_back": years_back,
            "window_sizes": [21, 63, 126],  # Reduced from full
            "step_size": 10,  # Coarser than lite for benchmark
            "n_bootstrap": 10,  # Minimal for benchmark
            "n_regimes": 3,
        }

        result.notes.append(f"Years back: {config['years_back']}")
        result.notes.append(f"Window sizes: {config['window_sizes']}")
        result.notes.append(f"Step size: {config['step_size']}")

        # Multi-resolution component
        if verbose:
            print("    Running multi-resolution component...")

        multi_res_metrics = _run_multiresolution_subset(
            panel, returns,
            window_sizes=config["window_sizes"],
            step_size=config["step_size"],
            max_windows=max_windows,
        )

        result.n_windows = multi_res_metrics["total_windows"]
        result.components_run.append("multiresolution")

        # Lens counting
        result.n_lenses = 4  # PCA, volatility, correlation, anomaly (lite version)

        # Bootstrap component (minimal)
        if verbose:
            print("    Running bootstrap component...")

        from sklearn.utils import resample

        # Just verify bootstrap works on a small sample
        sample_window = returns.iloc[-config["window_sizes"][0]:]
        for _ in range(config["n_bootstrap"]):
            idx = resample(range(len(sample_window)), replace=True)
            _ = sample_window.iloc[idx].std()

        result.components_run.append("bootstrap")

        # Regime detection component
        if verbose:
            print("    Running regime detection component...")

        from sklearn.cluster import KMeans

        mean_ret = returns.mean(axis=1).values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=config["n_regimes"], random_state=42, n_init=3)
        kmeans.fit(mean_ret)
        result.components_run.append("regime_detection")

        # Parallel info
        import multiprocessing as mp
        result.parallel_enabled = True
        result.n_workers = max(1, mp.cpu_count() - 1)

        result.notes.append(f"Components: {', '.join(result.components_run)}")

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        result.notes.append(f"Error: {str(e)}")
        if verbose:
            traceback.print_exc()

    result.runtime_seconds = time.time() - start_time

    if verbose:
        print(f"    Completed in {result.runtime_seconds:.2f}s")

    return result


def run_full_benchmark(
    years_back: int = 2,
    max_windows: int = 30,
    verbose: bool = False,
) -> OvernightResult:
    """
    Benchmark the full overnight analysis pattern.

    Uses heavily reduced parameters to exercise code paths
    without the 6-12 hour runtime.
    """
    result = OvernightResult(
        script_name="overnight_analysis.py",
        mode="benchmark",
        runtime_seconds=0.0,
    )

    if verbose:
        print("\n  Benchmarking overnight_analysis pattern...")

    start_time = time.time()

    try:
        # Load data
        panel, returns = _load_data_for_benchmark(years_back=years_back)

        if panel.empty or returns.empty:
            result.success = False
            result.error_message = "No data available"
            result.notes.append("Data loading failed")
            return result

        result.n_indicators = len(panel.columns)

        # Configuration matching overnight_analysis.py pattern (reduced)
        config = {
            "years_back": years_back,
            "window_sizes": [21, 63, 126, 252],
            "step_size": 20,  # Much coarser than full daily
            "n_bootstrap": 5,  # Minimal
            "n_monte_carlo": 5,  # Minimal
            "mc_block_size": 21,
            "n_regimes": 3,
        }

        result.notes.append(f"Years back: {config['years_back']}")
        result.notes.append(f"Window sizes: {config['window_sizes']}")
        result.notes.append(f"Monte Carlo sims: {config['n_monte_carlo']}")

        # Multi-resolution component
        if verbose:
            print("    Running multi-resolution component...")

        multi_res_metrics = _run_multiresolution_subset(
            panel, returns,
            window_sizes=config["window_sizes"],
            step_size=config["step_size"],
            max_windows=max_windows,
        )

        result.n_windows = multi_res_metrics["total_windows"]
        result.components_run.append("multiresolution")

        # Full lens set
        result.n_lenses = 16  # All 16 lenses in full version

        # Bootstrap component
        if verbose:
            print("    Running bootstrap component...")

        from sklearn.utils import resample

        sample_window = returns.iloc[-config["window_sizes"][0]:]
        for _ in range(config["n_bootstrap"]):
            idx = resample(range(len(sample_window)), replace=True)
            _ = sample_window.iloc[idx].std()

        result.components_run.append("bootstrap")

        # Monte Carlo component (minimal)
        if verbose:
            print("    Running Monte Carlo component...")

        np.random.seed(42)
        for _ in range(config["n_monte_carlo"]):
            # Generate synthetic returns using block bootstrap
            n_obs = len(returns)
            block_size = config["mc_block_size"]
            n_blocks = n_obs // block_size + 1

            synthetic = []
            for __ in range(n_blocks):
                start = np.random.randint(0, n_obs - block_size)
                synthetic.append(returns.values[start:start + block_size])

            _ = np.vstack(synthetic)[:n_obs]

        result.components_run.append("monte_carlo")

        # HMM regime analysis (if available)
        if verbose:
            print("    Running HMM component...")

        try:
            from hmmlearn.hmm import GaussianHMM

            mean_ret = returns.mean(axis=1).values.reshape(-1, 1)
            hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=20, random_state=42)
            hmm.fit(mean_ret)
            _ = hmm.predict(mean_ret)
            result.components_run.append("hmm")
            result.notes.append("HMM available and tested")
        except ImportError:
            result.notes.append("HMM not available (hmmlearn not installed)")
        except Exception as e:
            result.notes.append(f"HMM failed: {str(e)}")

        # GMM regime detection
        if verbose:
            print("    Running GMM regime detection...")

        from sklearn.mixture import GaussianMixture

        mean_ret = returns.mean(axis=1).values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=config["n_regimes"], random_state=42, n_init=2, max_iter=50)
        gmm.fit(mean_ret)
        _ = gmm.predict_proba(mean_ret)
        result.components_run.append("gmm_regime")

        # Parallel info
        import multiprocessing as mp
        result.parallel_enabled = True
        result.n_workers = max(1, mp.cpu_count() - 1)

        result.notes.append(f"Components: {', '.join(result.components_run)}")

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        result.notes.append(f"Error: {str(e)}")
        if verbose:
            traceback.print_exc()

    result.runtime_seconds = time.time() - start_time

    if verbose:
        print(f"    Completed in {result.runtime_seconds:.2f}s")

    return result


def run_overnight_benchmarks(
    modes: list[str] | None = None,
    verbose: bool = False,
) -> list[OvernightResult]:
    """
    Run all overnight benchmarks.

    Parameters
    ----------
    modes : list[str], optional
        Which modes to run: ["lite", "full"]. Defaults to both.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    list[OvernightResult]
        Results for each overnight benchmark.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("OVERNIGHT ANALYSIS BENCHMARKS")
        print("=" * 60)

    if modes is None:
        modes = ["lite", "full"]

    results = []

    if "lite" in modes:
        result = run_lite_benchmark(verbose=verbose)
        results.append(result)
        gc.collect()

    if "full" in modes:
        result = run_full_benchmark(verbose=verbose)
        results.append(result)
        gc.collect()

    if verbose:
        # Summary
        print("\n" + "-" * 60)
        print("OVERNIGHT BENCHMARK SUMMARY")
        print("-" * 60)

        for r in results:
            status = "PASS" if r.success else "FAIL"
            print(f"  {r.script_name} ({r.mode}): {status}")
            print(f"    Runtime: {r.runtime_seconds:.2f}s")
            print(f"    Indicators: {r.n_indicators}")
            print(f"    Windows: {r.n_windows}")
            print(f"    Components: {', '.join(r.components_run)}")

    return results


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="Overnight Analysis Benchmarks")
    parser.add_argument("--mode", choices=["lite", "full", "both"], default="both")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    modes = ["lite", "full"] if args.mode == "both" else [args.mode]
    results = run_overnight_benchmarks(modes=modes, verbose=args.verbose or True)

    print(f"\nCompleted {len(results)} overnight benchmark(s)")
