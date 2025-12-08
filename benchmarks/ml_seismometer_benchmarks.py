"""
ML / Seismometer Benchmarks for PRISM Engine

Benchmarks for the ML components including:
- Seismometer ensemble (instability detection)
- Regime classification
- Individual detector components

These benchmarks test that ML components run correctly and measure
their performance. Correctness evaluation is light (no model quality
assessment) - we verify the pipeline runs and produces output.
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
class MLBenchmarkResult:
    """Result from an ML/Seismometer benchmark."""

    component: str  # e.g., "seismometer", "regime_ensemble"
    runtime_seconds: float
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    output_snapshot: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    subcomponents_tested: list[str] = field(default_factory=list)


def _load_features_for_benchmark(
    years_back: int = 3,
) -> pd.DataFrame:
    """Load feature data for ML benchmarks."""
    from data.sql.db_connector import load_all_indicators_wide

    panel = load_all_indicators_wide()

    if panel.empty:
        return pd.DataFrame()

    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    # Filter to analysis period
    cutoff = panel.index.max() - pd.DateOffset(years=years_back)
    panel = panel[panel.index >= cutoff]

    # Keep columns with 50%+ coverage
    valid_cols = panel.columns[panel.notna().sum() > len(panel) * 0.5]
    panel = panel[valid_cols].ffill().bfill().dropna(axis=1)

    return panel


def run_seismometer_benchmark(
    years_back: int = 3,
    baseline_years: int = 2,
    verbose: bool = False,
) -> MLBenchmarkResult:
    """
    Benchmark the Seismometer ensemble.

    Tests the full seismometer pipeline:
    1. Load features
    2. Fit on baseline period
    3. Score all observations
    4. Get stability assessment
    """
    result = MLBenchmarkResult(
        component="seismometer",
        runtime_seconds=0.0,
    )

    if verbose:
        print("\n  Benchmarking Seismometer...")

    start_time = time.time()

    try:
        # Load features
        if verbose:
            print("    Loading features...")

        features = _load_features_for_benchmark(years_back=years_back)

        if features.empty:
            result.success = False
            result.error_message = "No features available"
            result.notes.append("Feature loading failed")
            return result

        result.n_samples = len(features)
        result.n_features = len(features.columns)

        if verbose:
            print(f"    Loaded {result.n_samples} samples, {result.n_features} features")

        # Define baseline period
        end_date = features.index.max()
        baseline_end = end_date - pd.DateOffset(years=years_back - baseline_years)
        baseline_start = baseline_end - pd.DateOffset(years=baseline_years)

        result.notes.append(f"Baseline: {baseline_start.date()} to {baseline_end.date()}")

        # Initialize seismometer
        if verbose:
            print("    Initializing Seismometer...")

        from engine_core.seismometer.ensemble import Seismometer

        seismo = Seismometer(
            n_clusters=5,
            encoding_dim=min(8, result.n_features - 1),  # Ensure valid encoding dim
            correlation_window=60,
        )

        result.subcomponents_tested.append("initialization")

        # Fit on baseline period
        if verbose:
            print("    Fitting on baseline period...")

        fit_start = time.time()
        seismo.fit(
            features=features,
            start=str(baseline_start.date()),
            end=str(baseline_end.date()),
        )
        fit_time = time.time() - fit_start

        result.notes.append(f"Fit time: {fit_time:.2f}s")
        result.subcomponents_tested.append("fit")

        # Score all observations
        if verbose:
            print("    Scoring observations...")

        score_start = time.time()
        scores = seismo.score(features)
        score_time = time.time() - score_start

        result.notes.append(f"Score time: {score_time:.2f}s")
        result.subcomponents_tested.append("score")

        # Get current status
        if verbose:
            print("    Getting status...")

        status = seismo.get_current_status()
        result.subcomponents_tested.append("status")

        # Capture output snapshot
        result.output_snapshot = {
            "stability_index": status.get("stability_index"),
            "alert_level": status.get("alert_level"),
            "date": status.get("date"),
            "n_scored_samples": len(scores),
            "mean_stability": float(scores["stability_index"].mean()),
            "min_stability": float(scores["stability_index"].min()),
            "max_stability": float(scores["stability_index"].max()),
            "components": status.get("components", {}),
        }

        # Test alert history
        alert_history = seismo.get_alert_history()
        result.output_snapshot["n_alert_levels"] = alert_history["alert_level"].nunique()
        result.subcomponents_tested.append("alert_history")

        if verbose:
            print(f"    Current status: {status.get('alert_level')} "
                  f"(stability: {status.get('stability_index', 0):.3f})")

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


def run_regime_classification_benchmark(
    years_back: int = 3,
    n_regimes: int = 3,
    verbose: bool = False,
) -> MLBenchmarkResult:
    """
    Benchmark regime classification models.

    Tests GMM-based and K-means regime detection.
    """
    result = MLBenchmarkResult(
        component="regime_classification",
        runtime_seconds=0.0,
    )

    if verbose:
        print("\n  Benchmarking Regime Classification...")

    start_time = time.time()

    try:
        # Load features
        features = _load_features_for_benchmark(years_back=years_back)

        if features.empty:
            result.success = False
            result.error_message = "No features available"
            return result

        result.n_samples = len(features)
        result.n_features = len(features.columns)

        # Compute returns for regime detection
        returns = features.pct_change().dropna()
        mean_ret = returns.mean(axis=1).values.reshape(-1, 1)

        # GMM regime detection
        if verbose:
            print("    Testing GMM regime detection...")

        from sklearn.mixture import GaussianMixture

        gmm_start = time.time()
        gmm = GaussianMixture(
            n_components=n_regimes,
            random_state=42,
            n_init=3,
            max_iter=100,
        )
        gmm.fit(mean_ret)
        gmm_labels = gmm.predict(mean_ret)
        gmm_probs = gmm.predict_proba(mean_ret)
        gmm_time = time.time() - gmm_start

        result.subcomponents_tested.append("gmm")
        result.notes.append(f"GMM fit time: {gmm_time:.2f}s")

        # K-means regime detection
        if verbose:
            print("    Testing K-means regime detection...")

        from sklearn.cluster import KMeans

        kmeans_start = time.time()
        kmeans = KMeans(
            n_clusters=n_regimes,
            random_state=42,
            n_init=5,
        )
        kmeans.fit(mean_ret)
        kmeans_labels = kmeans.predict(mean_ret)
        kmeans_time = time.time() - kmeans_start

        result.subcomponents_tested.append("kmeans")
        result.notes.append(f"K-means fit time: {kmeans_time:.2f}s")

        # HMM regime detection (if available)
        hmm_available = False
        try:
            from hmmlearn.hmm import GaussianHMM

            if verbose:
                print("    Testing HMM regime detection...")

            hmm_start = time.time()
            hmm = GaussianHMM(
                n_components=min(2, n_regimes),  # HMM works better with fewer states
                covariance_type="diag",
                n_iter=50,
                random_state=42,
            )
            hmm.fit(mean_ret)
            hmm_labels = hmm.predict(mean_ret)
            hmm_time = time.time() - hmm_start

            result.subcomponents_tested.append("hmm")
            result.notes.append(f"HMM fit time: {hmm_time:.2f}s")
            hmm_available = True
        except ImportError:
            result.notes.append("HMM not available (hmmlearn not installed)")
        except Exception as e:
            result.notes.append(f"HMM failed: {str(e)}")

        # Capture output snapshot
        result.output_snapshot = {
            "n_regimes": n_regimes,
            "gmm_labels_unique": int(np.unique(gmm_labels).shape[0]),
            "kmeans_labels_unique": int(np.unique(kmeans_labels).shape[0]),
            "gmm_current_regime": int(gmm_labels[-1]),
            "gmm_current_probs": gmm_probs[-1].tolist(),
            "kmeans_current_regime": int(kmeans_labels[-1]),
            "gmm_regime_distribution": {
                int(k): int(v) for k, v in zip(
                    *np.unique(gmm_labels, return_counts=True)
                )
            },
        }

        if hmm_available:
            result.output_snapshot["hmm_current_regime"] = int(hmm_labels[-1])
            result.output_snapshot["hmm_labels_unique"] = int(np.unique(hmm_labels).shape[0])

        if verbose:
            print(f"    Current GMM regime: {gmm_labels[-1]}")
            print(f"    Current K-means regime: {kmeans_labels[-1]}")

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


def run_detector_components_benchmark(
    years_back: int = 2,
    verbose: bool = False,
) -> MLBenchmarkResult:
    """
    Benchmark individual seismometer detector components.

    Tests each detector in isolation:
    - ClusteringDriftDetector
    - ReconstructionErrorDetector
    - CorrelationGraphDetector
    """
    result = MLBenchmarkResult(
        component="detector_components",
        runtime_seconds=0.0,
    )

    if verbose:
        print("\n  Benchmarking Detector Components...")

    start_time = time.time()

    try:
        # Load features
        features = _load_features_for_benchmark(years_back=years_back)

        if features.empty:
            result.success = False
            result.error_message = "No features available"
            return result

        result.n_samples = len(features)
        result.n_features = len(features.columns)

        # Split into baseline and test
        split_idx = len(features) // 2
        baseline = features.iloc[:split_idx]
        test_data = features.iloc[split_idx:]

        detector_times = {}

        # Test ClusteringDriftDetector
        if verbose:
            print("    Testing ClusteringDriftDetector...")

        try:
            from engine_core.seismometer.clustering import ClusteringDriftDetector

            det_start = time.time()
            clustering_det = ClusteringDriftDetector(n_clusters=5)
            clustering_det.fit(baseline)
            clustering_scores = clustering_det.score(test_data)
            detector_times["clustering_drift"] = time.time() - det_start

            result.subcomponents_tested.append("clustering_drift")
            result.output_snapshot["clustering_drift_mean"] = float(clustering_scores.mean())
        except Exception as e:
            result.notes.append(f"ClusteringDriftDetector failed: {str(e)}")

        # Test ReconstructionErrorDetector
        if verbose:
            print("    Testing ReconstructionErrorDetector...")

        try:
            from engine_core.seismometer.autoencoder import ReconstructionErrorDetector

            det_start = time.time()
            recon_det = ReconstructionErrorDetector(
                encoding_dim=min(8, result.n_features - 1)
            )
            recon_det.fit(baseline)
            recon_scores = recon_det.score(test_data)
            detector_times["reconstruction_error"] = time.time() - det_start

            result.subcomponents_tested.append("reconstruction_error")
            result.output_snapshot["reconstruction_error_mean"] = float(recon_scores.mean())
        except Exception as e:
            result.notes.append(f"ReconstructionErrorDetector failed: {str(e)}")

        # Test CorrelationGraphDetector
        if verbose:
            print("    Testing CorrelationGraphDetector...")

        try:
            from engine_core.seismometer.correlation_graph import CorrelationGraphDetector

            det_start = time.time()
            corr_det = CorrelationGraphDetector(window_days=60)
            corr_det.fit(baseline)
            corr_scores = corr_det.score(test_data)
            detector_times["correlation_graph"] = time.time() - det_start

            result.subcomponents_tested.append("correlation_graph")
            result.output_snapshot["correlation_graph_mean"] = float(corr_scores.mean())
        except Exception as e:
            result.notes.append(f"CorrelationGraphDetector failed: {str(e)}")

        # Summary
        result.output_snapshot["detector_times"] = detector_times
        total_detector_time = sum(detector_times.values())
        result.notes.append(f"Total detector time: {total_detector_time:.2f}s")

        if verbose:
            for det_name, det_time in detector_times.items():
                print(f"    {det_name}: {det_time:.2f}s")

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


def run_ml_seismometer_benchmarks(
    components: list[str] | None = None,
    verbose: bool = False,
) -> list[MLBenchmarkResult]:
    """
    Run all ML/Seismometer benchmarks.

    Parameters
    ----------
    components : list[str], optional
        Which components to benchmark:
        ["seismometer", "regime", "detectors"].
        Defaults to all.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    list[MLBenchmarkResult]
        Results for all benchmarks.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("ML / SEISMOMETER BENCHMARKS")
        print("=" * 60)

    if components is None:
        components = ["seismometer", "regime", "detectors"]

    results = []

    if "seismometer" in components:
        result = run_seismometer_benchmark(verbose=verbose)
        results.append(result)
        gc.collect()

    if "regime" in components:
        result = run_regime_classification_benchmark(verbose=verbose)
        results.append(result)
        gc.collect()

    if "detectors" in components:
        result = run_detector_components_benchmark(verbose=verbose)
        results.append(result)
        gc.collect()

    if verbose:
        # Summary
        print("\n" + "-" * 60)
        print("ML/SEISMOMETER BENCHMARK SUMMARY")
        print("-" * 60)

        for r in results:
            status = "PASS" if r.success else "FAIL"
            print(f"  {r.component}: {status}")
            print(f"    Runtime: {r.runtime_seconds:.2f}s")
            print(f"    Samples: {r.n_samples}, Features: {r.n_features}")
            if r.subcomponents_tested:
                print(f"    Subcomponents: {', '.join(r.subcomponents_tested)}")

    return results


if __name__ == "__main__":
    # Quick test
    import argparse

    parser = argparse.ArgumentParser(description="ML/Seismometer Benchmarks")
    parser.add_argument(
        "--component",
        choices=["seismometer", "regime", "detectors", "all"],
        default="all",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.component == "all":
        components = None
    else:
        components = [args.component]

    results = run_ml_seismometer_benchmarks(
        components=components,
        verbose=args.verbose or True,
    )

    print(f"\nCompleted {len(results)} ML/seismometer benchmark(s)")
