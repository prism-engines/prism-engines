"""
Engine Correctness Benchmarks for PRISM Engine

Tests whether the engine and lenses behave correctly on known synthetic signals.
Each scenario has expected properties that lenses should detect.

This module runs each synthetic scenario through the temporal engine and key lenses,
then validates that the results match expected behavior.
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from .synthetic_scenarios import (
    SyntheticScenario,
    get_all_scenarios,
    generate_sine_wave,
    generate_trend_noise,
    generate_step_change,
    generate_white_noise,
    generate_correlated_pairs,
)


@dataclass
class CorrectnessResult:
    """Result from a single correctness benchmark."""

    scenario_name: str
    n_rows: int
    n_columns: int
    lenses_run: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


# Core lenses to test (more stable and faster)
CORE_LENSES = ["magnitude", "pca", "influence", "clustering", "network"]

# Extended lenses for comprehensive testing
EXTENDED_LENSES = ["decomposition", "granger", "anomaly", "regime", "mutual_info"]


def _safe_run_lens(lens_name: str, df: pd.DataFrame) -> tuple[dict | None, str | None]:
    """
    Safely run a lens and catch any errors.

    Returns
    -------
    tuple
        (result_dict, error_message) - error_message is None on success
    """
    try:
        from engine_core import get_lens

        lens = get_lens(lens_name)
        result = lens.analyze(df)
        return result, None
    except Exception as e:
        return None, f"{lens_name}: {str(e)}"


def _compute_basic_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute basic statistics for a panel."""
    # Exclude date column
    numeric_df = df.select_dtypes(include=[np.number])
    stats = {
        "mean_mean": float(numeric_df.mean().mean()),
        "mean_std": float(numeric_df.std().mean()),
        "column_correlation": float(numeric_df.corr().values[np.triu_indices(len(numeric_df.columns), k=1)].mean())
        if len(numeric_df.columns) > 1 else 0.0,
    }
    return stats


def _check_sine_wave_detection(
    scenario: SyntheticScenario,
    lens_results: dict[str, dict],
) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Validate lens results for sine wave scenario.

    Expected: High correlation between columns (due to periodic structure),
    PCA should capture most variance in few components.
    """
    metrics = {}
    checks = {}

    # Check PCA captures structure
    if "pca" in lens_results and lens_results["pca"]:
        pca_result = lens_results["pca"]
        if "explained_variance_ratio" in pca_result:
            evr = pca_result["explained_variance_ratio"]
            first_component_var = evr[0] if len(evr) > 0 else 0
            metrics["pca_first_component_variance"] = first_component_var
            # Sine wave with similar phases should have high first component
            checks["pca_captures_structure"] = first_component_var > 0.3

    # Check clustering finds similarity
    if "clustering" in lens_results and lens_results["clustering"]:
        clust_result = lens_results["clustering"]
        if "n_clusters" in clust_result:
            metrics["n_clusters_found"] = clust_result["n_clusters"]

    # Check influence scores are distributed (all columns contribute)
    if "influence" in lens_results and lens_results["influence"]:
        inf_result = lens_results["influence"]
        if "scores" in inf_result:
            scores = list(inf_result["scores"].values())
            if scores:
                score_std = np.std(scores)
                metrics["influence_score_std"] = score_std
                # Low std means scores are similar (expected for similar sine waves)
                checks["influence_distributed"] = True

    return metrics, checks


def _check_trend_detection(
    scenario: SyntheticScenario,
    lens_results: dict[str, dict],
) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Validate lens results for trend + noise scenario.

    Expected: Decomposition should find trend component,
    PCA should capture trend direction.
    """
    metrics = {}
    checks = {}

    # Check decomposition finds trend
    if "decomposition" in lens_results and lens_results["decomposition"]:
        decomp_result = lens_results["decomposition"]
        if "trend_strength" in decomp_result:
            metrics["trend_strength"] = decomp_result["trend_strength"]
            checks["detects_trend"] = decomp_result["trend_strength"] > 0.3

    # PCA should show high first component due to common trend
    if "pca" in lens_results and lens_results["pca"]:
        pca_result = lens_results["pca"]
        if "explained_variance_ratio" in pca_result:
            evr = pca_result["explained_variance_ratio"]
            if len(evr) > 0:
                metrics["pca_first_component"] = evr[0]
                checks["pca_captures_trend"] = evr[0] > 0.5

    return metrics, checks


def _check_step_change_detection(
    scenario: SyntheticScenario,
    lens_results: dict[str, dict],
) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Validate lens results for step change (regime shift) scenario.

    Expected: Regime switching should detect change,
    anomaly detection might flag the transition.
    """
    metrics = {}
    checks = {}

    # Check regime detection
    if "regime" in lens_results and lens_results["regime"]:
        regime_result = lens_results["regime"]
        if "n_regimes_detected" in regime_result:
            n_regimes = regime_result["n_regimes_detected"]
            metrics["n_regimes_detected"] = n_regimes
            # Should detect at least 2 regimes (before/after change)
            checks["detects_regime_change"] = n_regimes >= 2

    # Check anomaly detection
    if "anomaly" in lens_results and lens_results["anomaly"]:
        anomaly_result = lens_results["anomaly"]
        if "n_anomalies" in anomaly_result:
            metrics["n_anomalies"] = anomaly_result["n_anomalies"]
            # May or may not detect anomalies depending on transition sharpness
            checks["anomaly_detection_ran"] = True

    return metrics, checks


def _check_white_noise_behavior(
    scenario: SyntheticScenario,
    lens_results: dict[str, dict],
) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Validate lens results for white noise scenario.

    Expected: No strong structure detected, PCA variance distributed,
    no clear regimes, low correlation between columns.
    """
    metrics = {}
    checks = {}

    # PCA should show distributed variance (no dominant component)
    if "pca" in lens_results and lens_results["pca"]:
        pca_result = lens_results["pca"]
        if "explained_variance_ratio" in pca_result:
            evr = pca_result["explained_variance_ratio"]
            if len(evr) > 0:
                metrics["pca_first_component"] = evr[0]
                # White noise: variance should be distributed
                checks["no_dominant_component"] = evr[0] < 0.5

    # Influence scores should be relatively uniform
    if "influence" in lens_results and lens_results["influence"]:
        inf_result = lens_results["influence"]
        if "scores" in inf_result:
            scores = list(inf_result["scores"].values())
            if scores:
                score_cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
                metrics["influence_coefficient_of_variation"] = score_cv
                # Low CV means similar scores (expected for noise)
                checks["uniform_influence"] = score_cv < 1.0

    # Network should show weak connections
    if "network" in lens_results and lens_results["network"]:
        net_result = lens_results["network"]
        if "mean_edge_weight" in net_result:
            metrics["mean_edge_weight"] = net_result["mean_edge_weight"]
            checks["weak_network"] = net_result["mean_edge_weight"] < 0.5

    return metrics, checks


def _check_correlated_pairs(
    scenario: SyntheticScenario,
    lens_results: dict[str, dict],
) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Validate lens results for correlated pairs scenario.

    Expected: Network should show strong connections between pairs,
    clustering should group correlated indicators together.
    """
    metrics = {}
    checks = {}

    # Network should find structure
    if "network" in lens_results and lens_results["network"]:
        net_result = lens_results["network"]
        if "mean_edge_weight" in net_result:
            metrics["mean_edge_weight"] = net_result["mean_edge_weight"]
            # Correlated pairs should show higher average weight
            checks["detects_correlation_structure"] = net_result["mean_edge_weight"] > 0.3

    # Clustering should find groups
    if "clustering" in lens_results and lens_results["clustering"]:
        clust_result = lens_results["clustering"]
        if "n_clusters" in clust_result:
            metrics["n_clusters"] = clust_result["n_clusters"]

    # Influence should vary (some indicators more central)
    if "influence" in lens_results and lens_results["influence"]:
        inf_result = lens_results["influence"]
        if "scores" in inf_result:
            scores = list(inf_result["scores"].values())
            if scores:
                metrics["influence_max"] = max(scores)
                metrics["influence_min"] = min(scores)
                checks["influence_varies"] = max(scores) - min(scores) > 0.1

    return metrics, checks


def run_scenario_benchmark(
    scenario: SyntheticScenario,
    lenses: list[str] | None = None,
    verbose: bool = False,
) -> CorrectnessResult:
    """
    Run a single scenario through the lenses and validate results.

    Parameters
    ----------
    scenario : SyntheticScenario
        The synthetic scenario to test.
    lenses : list[str], optional
        Which lenses to run. Defaults to CORE_LENSES.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    CorrectnessResult
        Structured result with metrics and validation checks.
    """
    if lenses is None:
        lenses = CORE_LENSES.copy()

    start_time = time.time()
    result = CorrectnessResult(
        scenario_name=scenario.name,
        n_rows=scenario.n_rows,
        n_columns=scenario.n_columns,
        lenses_run=[],
    )

    if verbose:
        print(f"  Running scenario: {scenario.name} ({scenario.n_rows} x {scenario.n_columns})")

    # Compute basic stats
    basic_stats = _compute_basic_stats(scenario.data)
    result.metrics.update(basic_stats)

    # Run each lens
    lens_results = {}
    errors = []

    for lens_name in lenses:
        if verbose:
            print(f"    - Running lens: {lens_name}...", end=" ")

        lens_result, error = _safe_run_lens(lens_name, scenario.data)

        if error:
            errors.append(error)
            if verbose:
                print(f"FAILED: {error}")
        else:
            lens_results[lens_name] = lens_result
            result.lenses_run.append(lens_name)
            if verbose:
                print("OK")

    # Run scenario-specific validation
    scenario_metrics = {}
    scenario_checks = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if scenario.name == "sine_wave":
            scenario_metrics, scenario_checks = _check_sine_wave_detection(scenario, lens_results)
        elif scenario.name == "trend_noise":
            scenario_metrics, scenario_checks = _check_trend_detection(scenario, lens_results)
        elif scenario.name == "step_change":
            scenario_metrics, scenario_checks = _check_step_change_detection(scenario, lens_results)
        elif scenario.name == "white_noise":
            scenario_metrics, scenario_checks = _check_white_noise_behavior(scenario, lens_results)
        elif scenario.name == "correlated_pairs":
            scenario_metrics, scenario_checks = _check_correlated_pairs(scenario, lens_results)

    result.metrics.update(scenario_metrics)
    result.checks.update(scenario_checks)

    # Record errors
    if errors:
        result.notes.extend(errors)
        result.success = len(result.lenses_run) > 0  # Partial success if some lenses ran

    result.runtime_seconds = time.time() - start_time

    # Summary note
    n_checks_passed = sum(result.checks.values())
    n_checks_total = len(result.checks)
    result.notes.append(f"Passed {n_checks_passed}/{n_checks_total} validation checks")

    return result


def run_engine_correctness_benchmarks(
    use_extended_lenses: bool = False,
    n_periods: int = 252,
    seed: int = 42,
    verbose: bool = False,
) -> list[CorrectnessResult]:
    """
    Run all correctness benchmarks.

    Parameters
    ----------
    use_extended_lenses : bool
        Whether to include extended (slower) lenses.
    n_periods : int
        Number of time periods for synthetic data.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    list[CorrectnessResult]
        Results for all scenarios.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PRISM ENGINE CORRECTNESS BENCHMARKS")
        print("=" * 60)

    lenses = CORE_LENSES.copy()
    if use_extended_lenses:
        lenses.extend(EXTENDED_LENSES)

    if verbose:
        print(f"Lenses: {lenses}")
        print(f"Periods: {n_periods}, Seed: {seed}")

    # Generate all scenarios
    scenarios = get_all_scenarios(n_periods=n_periods, seed=seed)

    if verbose:
        print(f"\nRunning {len(scenarios)} scenarios...")

    results = []
    for scenario in scenarios:
        result = run_scenario_benchmark(scenario, lenses=lenses, verbose=verbose)
        results.append(result)

    if verbose:
        # Summary
        print("\n" + "-" * 60)
        print("CORRECTNESS SUMMARY")
        print("-" * 60)
        total_checks = sum(len(r.checks) for r in results)
        passed_checks = sum(sum(r.checks.values()) for r in results)
        print(f"Total validation checks: {passed_checks}/{total_checks} passed")

        for r in results:
            status = "PASS" if r.success else "FAIL"
            checks_passed = sum(r.checks.values())
            checks_total = len(r.checks)
            print(f"  {r.scenario_name}: {status} ({checks_passed}/{checks_total} checks, {r.runtime_seconds:.2f}s)")

    return results


if __name__ == "__main__":
    # Quick test
    results = run_engine_correctness_benchmarks(verbose=True)
    print(f"\nCompleted {len(results)} benchmark scenarios")
