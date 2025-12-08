#!/usr/bin/env python3
"""
PRISM Benchmark Suite - Full Test & Reporting Pipeline
=======================================================

Runs the PRISM engine across all 6 synthetic benchmark datasets:
  1. clear_leader   - One indicator dominates
  2. two_regimes    - Two distinct market regimes
  3. clusters       - Three correlated clusters
  4. periodic       - Sinusoidal with harmonics
  5. anomalies      - Random spikes/outliers
  6. pure_noise     - No structure (null baseline)

For each dataset, the suite:
  - Runs PRISM analysis with 63-day window
  - Computes MRF, Lens Agreement, PCA geometry
  - Generates diagnostic plots
  - Produces a comprehensive Markdown report
  - Exports everything as a timestamped ZIP

Usage:
    python prism_run.py --benchmarks

    # Or directly:
    python workflows/benchmark_suite.py
"""

import json
import shutil
import sqlite3
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sql.db_connector import get_connection
from data.sql.db_path import get_db_path


# =============================================================================
# CONFIGURATION
# =============================================================================

BENCHMARK_GROUPS = {
    "clear_leader": {
        "description": "One dominant indicator with subordinate followers",
        "expected_structure": "Single indicator should rank #1 consistently across all lenses",
        "failure_modes": ["Leader not detected", "False positives in followers"],
    },
    "two_regimes": {
        "description": "Two distinct market regimes with different correlation structures",
        "expected_structure": "Regime-switching lenses should detect transition points",
        "failure_modes": ["Regime boundaries missed", "Over-segmentation"],
    },
    "clusters": {
        "description": "Three correlated clusters of indicators",
        "expected_structure": "Clustering lens should identify 3 groups; network lens shows dense subgraphs",
        "failure_modes": ["Cluster merging", "Spurious cluster detection"],
    },
    "periodic": {
        "description": "Sinusoidal signals with harmonics",
        "expected_structure": "Wavelet/spectral lenses detect periodic components; PCA captures oscillation",
        "failure_modes": ["Frequency aliasing", "Phase drift not captured"],
    },
    "anomalies": {
        "description": "Random spikes and outliers embedded in normal behavior",
        "expected_structure": "Anomaly lens should flag outlier points; magnitude lens spikes",
        "failure_modes": ["Missed anomalies", "False positive rate too high"],
    },
    "pure_noise": {
        "description": "Pure white noise with no structure (null baseline)",
        "expected_structure": "No lens should show persistent structure; uniform rankings expected",
        "failure_modes": ["Spurious structure detected", "Overfitting to noise"],
    },
}

ANALYSIS_CONFIG = {
    "window_days": 63,
    "step_days": 21,
    "lenses": ["magnitude", "pca", "influence", "clustering", "network", "anomaly"],
    "min_data_points": 50,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DatasetAnalysis:
    """Results from analyzing a single benchmark dataset."""
    name: str
    description: str
    n_indicators: int
    n_observations: int
    date_range: Tuple[str, str]

    # Lens results
    lens_rankings: Dict[str, List[Dict]] = field(default_factory=dict)
    lens_metrics: Dict[str, float] = field(default_factory=dict)

    # Aggregate metrics
    mrf_score: float = 0.0  # Mean Rank Fluctuation
    lens_agreement: float = 0.0  # Cross-lens Kendall's tau
    pca_explained_variance: List[float] = field(default_factory=list)
    top_indicators: List[str] = field(default_factory=list)

    # Interpretation
    structure_detected: bool = False
    interpretation: str = ""
    confidence: str = "low"  # low, medium, high

    # Runtime
    runtime_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class BenchmarkSuiteResults:
    """Aggregate results from the full benchmark suite."""
    timestamp: str
    output_dir: str
    datasets_analyzed: int = 0
    total_runtime_seconds: float = 0.0

    results: Dict[str, DatasetAnalysis] = field(default_factory=dict)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

    report_path: Optional[str] = None
    zip_path: Optional[str] = None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_benchmark_data(group_name: str) -> pd.DataFrame:
    """
    Load benchmark dataset from database.

    Uses Schema v2 - indicator_values table.

    Args:
        group_name: One of the BENCHMARK_GROUPS keys

    Returns:
        DataFrame with date column and indicator columns
    """
    conn = get_connection()

    # Query benchmark indicators for this group from indicator_values
    # Indicator names follow pattern: "{group_name}_{column}"
    query = """
        SELECT iv.indicator_name, iv.date, iv.value
        FROM indicator_values iv
        JOIN indicators i ON iv.indicator_name = i.indicator_name
        WHERE i.system = 'benchmark'
          AND iv.indicator_name LIKE ?
        ORDER BY iv.indicator_name, iv.date
    """

    df = pd.read_sql(query, conn, params=(f"{group_name}_%",))
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for benchmark group: {group_name}")

    # Pivot to wide format
    df_wide = df.pivot(index="date", columns="indicator_name", values="value")
    df_wide = df_wide.reset_index()
    df_wide.columns.name = None

    # Ensure date is datetime
    df_wide["date"] = pd.to_datetime(df_wide["date"])
    df_wide = df_wide.sort_values("date").reset_index(drop=True)

    return df_wide


def get_available_benchmark_groups() -> List[str]:
    """Return list of benchmark groups available in database.

    Uses Schema v2 - parses group names from indicator name prefixes.
    """
    conn = get_connection()

    # Get unique prefixes from benchmark indicator names
    # Names follow pattern: "{group_name}_{column}"
    query = """
        SELECT DISTINCT indicator_name
        FROM indicators
        WHERE system = 'benchmark'
        ORDER BY indicator_name
    """

    result = conn.execute(query).fetchall()
    conn.close()

    # Extract unique group prefixes
    groups = set()
    for row in result:
        name = row[0]
        # Split on last underscore to get group prefix
        # e.g., "clear_leader_A" -> "clear_leader"
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            groups.add(parts[0])

    return sorted(list(groups))


# =============================================================================
# ANALYSIS ENGINE
# =============================================================================

def run_lenses_on_data(df: pd.DataFrame, lenses: List[str]) -> Dict[str, Any]:
    """
    Run specified lenses on a DataFrame.

    Args:
        df: DataFrame with 'date' column and indicator columns
        lenses: List of lens names to run

    Returns:
        Dictionary with lens results and rankings
    """
    from engine_core import get_lens

    results = {}

    for lens_name in lenses:
        try:
            lens = get_lens(lens_name)
            analysis = lens.analyze(df)
            ranking = lens.rank_indicators(df)

            results[lens_name] = {
                "analysis": analysis,
                "ranking": ranking.to_dict("records") if isinstance(ranking, pd.DataFrame) else ranking,
                "success": True,
            }
        except Exception as e:
            results[lens_name] = {
                "analysis": {},
                "ranking": [],
                "success": False,
                "error": str(e),
            }

    return results


def compute_mrf(lens_rankings: Dict[str, List[Dict]]) -> float:
    """
    Compute Mean Rank Fluctuation across lenses.

    MRF measures how much indicator rankings vary across different lenses.
    Lower MRF = more agreement between lenses.
    """
    if not lens_rankings:
        return 0.0

    # Get all indicators
    all_indicators = set()
    for lens_results in lens_rankings.values():
        if isinstance(lens_results, list):
            for item in lens_results:
                if isinstance(item, dict) and "indicator" in item:
                    all_indicators.add(item["indicator"])

    if not all_indicators:
        return 0.0

    # Compute rank variance per indicator
    rank_variances = []

    for indicator in all_indicators:
        ranks = []
        for lens_name, lens_results in lens_rankings.items():
            if isinstance(lens_results, list):
                for item in lens_results:
                    if isinstance(item, dict) and item.get("indicator") == indicator:
                        rank = item.get("rank", item.get("score", 0))
                        if rank is not None:
                            ranks.append(float(rank))

        if len(ranks) > 1:
            rank_variances.append(np.std(ranks))

    return float(np.mean(rank_variances)) if rank_variances else 0.0


def compute_lens_agreement(lens_rankings: Dict[str, List[Dict]]) -> float:
    """
    Compute pairwise Kendall's tau between lens rankings.

    Returns average tau across all lens pairs.
    """
    from scipy.stats import kendalltau

    lens_names = [k for k, v in lens_rankings.items() if isinstance(v, list) and len(v) > 0]

    if len(lens_names) < 2:
        return 0.0

    # Build ranking vectors per lens
    all_indicators = set()
    for lens_name in lens_names:
        for item in lens_rankings[lens_name]:
            if isinstance(item, dict) and "indicator" in item:
                all_indicators.add(item["indicator"])

    all_indicators = sorted(all_indicators)

    if len(all_indicators) < 2:
        return 0.0

    # Create rank vectors
    rank_vectors = {}
    for lens_name in lens_names:
        rank_map = {}
        for item in lens_rankings[lens_name]:
            if isinstance(item, dict):
                ind = item.get("indicator")
                rank = item.get("rank", item.get("score", 0))
                if ind and rank is not None:
                    rank_map[ind] = float(rank)

        # Fill missing with worst rank
        max_rank = max(rank_map.values()) if rank_map else len(all_indicators)
        rank_vectors[lens_name] = [rank_map.get(ind, max_rank + 1) for ind in all_indicators]

    # Compute pairwise tau
    taus = []
    for i, lens1 in enumerate(lens_names):
        for lens2 in lens_names[i+1:]:
            try:
                tau, _ = kendalltau(rank_vectors[lens1], rank_vectors[lens2])
                if not np.isnan(tau):
                    taus.append(tau)
            except Exception:
                pass

    return float(np.mean(taus)) if taus else 0.0


def analyze_dataset(group_name: str, config: Dict) -> DatasetAnalysis:
    """
    Run full PRISM analysis on a benchmark dataset.

    Args:
        group_name: Benchmark group name
        config: Analysis configuration

    Returns:
        DatasetAnalysis with all results
    """
    start_time = time.time()

    group_info = BENCHMARK_GROUPS.get(group_name, {})

    result = DatasetAnalysis(
        name=group_name,
        description=group_info.get("description", ""),
        n_indicators=0,
        n_observations=0,
        date_range=("", ""),
    )

    try:
        # Load data
        df = load_benchmark_data(group_name)

        indicator_cols = [c for c in df.columns if c != "date"]
        result.n_indicators = len(indicator_cols)
        result.n_observations = len(df)
        result.date_range = (str(df["date"].min().date()), str(df["date"].max().date()))

        if result.n_observations < config.get("min_data_points", 50):
            result.success = False
            result.error_message = f"Insufficient data: {result.n_observations} rows"
            return result

        # Run lenses
        lens_results = run_lenses_on_data(df, config.get("lenses", ["pca", "magnitude"]))

        # Extract rankings
        for lens_name, lr in lens_results.items():
            if lr.get("success"):
                result.lens_rankings[lens_name] = lr.get("ranking", [])

                # Extract key metrics
                analysis = lr.get("analysis", {})
                if "explained_variance_ratio" in analysis:
                    result.pca_explained_variance = list(analysis["explained_variance_ratio"][:5])
                if "scores" in analysis:
                    result.lens_metrics[f"{lens_name}_mean_score"] = float(np.mean(list(analysis["scores"].values())))

        # Compute aggregate metrics
        result.mrf_score = compute_mrf(result.lens_rankings)
        result.lens_agreement = compute_lens_agreement(result.lens_rankings)

        # Get top indicators (consensus)
        indicator_total_rank = {}
        for lens_rankings in result.lens_rankings.values():
            if isinstance(lens_rankings, list):
                for item in lens_rankings:
                    if isinstance(item, dict):
                        ind = item.get("indicator")
                        rank = item.get("rank", item.get("score", 0))
                        if ind and rank is not None:
                            indicator_total_rank[ind] = indicator_total_rank.get(ind, 0) + float(rank)

        sorted_indicators = sorted(indicator_total_rank.items(), key=lambda x: x[1])
        result.top_indicators = [ind for ind, _ in sorted_indicators[:5]]

        # Determine if structure was detected
        result.structure_detected = (
            result.lens_agreement > 0.3 or
            (result.pca_explained_variance and result.pca_explained_variance[0] > 0.4) or
            result.mrf_score < 2.0
        )

        # Generate interpretation
        result.interpretation = _generate_interpretation(result, group_info)

        # Confidence level
        if result.lens_agreement > 0.6:
            result.confidence = "high"
        elif result.lens_agreement > 0.3:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        result.success = True

    except Exception as e:
        result.success = False
        result.error_message = str(e)

    result.runtime_seconds = time.time() - start_time

    return result


def _generate_interpretation(result: DatasetAnalysis, group_info: Dict) -> str:
    """Generate natural language interpretation of results."""
    lines = []

    if result.structure_detected:
        lines.append(f"Structure detected in {result.name} dataset.")
    else:
        lines.append(f"No significant structure detected in {result.name} dataset.")

    if result.pca_explained_variance:
        pc1 = result.pca_explained_variance[0] * 100
        lines.append(f"First principal component explains {pc1:.1f}% of variance.")

    if result.lens_agreement > 0.5:
        lines.append(f"High lens agreement (tau={result.lens_agreement:.2f}) suggests robust signal.")
    elif result.lens_agreement < 0.2:
        lines.append(f"Low lens agreement (tau={result.lens_agreement:.2f}) indicates ambiguous structure.")

    if result.top_indicators:
        lines.append(f"Top-ranked indicators: {', '.join(result.top_indicators[:3])}")

    expected = group_info.get("expected_structure", "")
    if expected:
        lines.append(f"Expected: {expected}")

    return " ".join(lines)


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_dataset_plots(
    group_name: str,
    result: DatasetAnalysis,
    output_dir: Path,
) -> List[str]:
    """
    Generate diagnostic plots for a dataset analysis.

    Returns list of generated plot filenames.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = []

    try:
        # Load data for plotting
        df = load_benchmark_data(group_name)
        indicator_cols = [c for c in df.columns if c != "date"]

        # 1. Time Series Plot
        fig, ax = plt.subplots(figsize=(14, 6))
        for col in indicator_cols[:6]:  # Limit to 6 for readability
            ax.plot(df["date"], df[col], label=col, alpha=0.7)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(f"{group_name}: Time Series Overview")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        plot_path = output_dir / f"{group_name}_timeseries.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        plots.append(plot_path.name)

        # 2. PCA Scatter Plot
        if len(indicator_cols) >= 2:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            numeric_data = df[indicator_cols].dropna()
            if len(numeric_data) > 10:
                scaled = StandardScaler().fit_transform(numeric_data)
                pca = PCA(n_components=min(2, len(indicator_cols)))
                components = pca.fit_transform(scaled)

                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(
                    components[:, 0],
                    components[:, 1] if components.shape[1] > 1 else np.zeros(len(components)),
                    c=range(len(components)),
                    cmap="viridis",
                    alpha=0.6,
                )
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)" if len(pca.explained_variance_ratio_) > 1 else "")
                ax.set_title(f"{group_name}: PCA Projection")
                plt.colorbar(scatter, label="Time Index")

                plot_path = output_dir / f"{group_name}_pca_scatter.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                plots.append(plot_path.name)

        # 3. Correlation Heatmap
        if len(indicator_cols) >= 2:
            corr = df[indicator_cols].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(indicator_cols)))
            ax.set_yticks(range(len(indicator_cols)))
            ax.set_xticklabels(indicator_cols, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(indicator_cols, fontsize=8)
            ax.set_title(f"{group_name}: Correlation Matrix")
            plt.colorbar(im, label="Correlation")

            plot_path = output_dir / f"{group_name}_correlation.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            plots.append(plot_path.name)

        # 4. Lens Agreement Chart
        if result.lens_rankings:
            fig, ax = plt.subplots(figsize=(12, 6))

            lens_names = list(result.lens_rankings.keys())
            x = np.arange(len(lens_names))

            # Get mean rank for top 3 indicators per lens
            for i, (lens, rankings) in enumerate(result.lens_rankings.items()):
                if isinstance(rankings, list) and len(rankings) > 0:
                    top3_ranks = [r.get("rank", r.get("score", 0)) for r in rankings[:3]]
                    ax.bar(i, np.mean(top3_ranks) if top3_ranks else 0, alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels(lens_names, rotation=45, ha="right")
            ax.set_ylabel("Mean Top-3 Rank")
            ax.set_title(f"{group_name}: Lens Ranking Comparison")
            ax.grid(True, alpha=0.3, axis="y")

            plot_path = output_dir / f"{group_name}_lens_agreement.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            plots.append(plot_path.name)

        # 5. MRF Timeline (rolling rank volatility)
        if len(df) > ANALYSIS_CONFIG["window_days"]:
            window = ANALYSIS_CONFIG["window_days"]
            rolling_std = df[indicator_cols].rolling(window=window).std().mean(axis=1)

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.fill_between(df["date"], rolling_std, alpha=0.5, label="Volatility")
            ax.plot(df["date"], rolling_std, color="darkblue", linewidth=1)
            ax.set_xlabel("Date")
            ax.set_ylabel("Rolling Volatility")
            ax.set_title(f"{group_name}: Mean Rank Fluctuation Timeline ({window}-day window)")
            ax.grid(True, alpha=0.3)

            plot_path = output_dir / f"{group_name}_mrf_timeline.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            plots.append(plot_path.name)

    except Exception as e:
        print(f"Warning: Error generating plots for {group_name}: {e}")

    return plots


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(results: BenchmarkSuiteResults) -> str:
    """Generate comprehensive Markdown report."""
    lines = []

    # Header
    lines.append("# PRISM Engine Benchmark Suite Report")
    lines.append("")
    lines.append(f"**Generated**: {results.timestamp}")
    lines.append(f"**Datasets Analyzed**: {results.datasets_analyzed}")
    lines.append(f"**Total Runtime**: {results.total_runtime_seconds:.1f} seconds")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report presents the results of running the PRISM analytical engine across")
    lines.append("six synthetic benchmark datasets with known statistical properties. Each dataset")
    lines.append("tests specific aspects of the engine's detection capabilities.")
    lines.append("")

    # Summary Table
    lines.append("### Quick Summary")
    lines.append("")
    lines.append("| Dataset | Indicators | Observations | Lens Agreement | Structure Detected | Confidence |")
    lines.append("|---------|------------|--------------|----------------|-------------------|------------|")

    for name, result in results.results.items():
        detected = "Yes" if result.structure_detected else "No"
        lines.append(
            f"| {name} | {result.n_indicators} | {result.n_observations} | "
            f"{result.lens_agreement:.2f} | {detected} | {result.confidence} |"
        )
    lines.append("")

    # Detailed Results
    lines.append("## Detailed Results")
    lines.append("")

    for name, result in results.results.items():
        group_info = BENCHMARK_GROUPS.get(name, {})

        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"**Description**: {result.description}")
        lines.append("")

        # Dataset stats
        lines.append("#### Dataset Statistics")
        lines.append("")
        lines.append(f"- **Indicators**: {result.n_indicators}")
        lines.append(f"- **Observations**: {result.n_observations}")
        lines.append(f"- **Date Range**: {result.date_range[0]} to {result.date_range[1]}")
        lines.append(f"- **Runtime**: {result.runtime_seconds:.2f}s")
        lines.append("")

        # Expected structure
        lines.append("#### Expected Structure")
        lines.append("")
        lines.append(group_info.get("expected_structure", "Not specified"))
        lines.append("")

        # Engine outputs
        lines.append("#### Engine Outputs")
        lines.append("")
        lines.append(f"- **Lens Agreement (Kendall's tau)**: {result.lens_agreement:.3f}")
        lines.append(f"- **Mean Rank Fluctuation**: {result.mrf_score:.3f}")

        if result.pca_explained_variance:
            pca_str = ", ".join([f"{v*100:.1f}%" for v in result.pca_explained_variance[:3]])
            lines.append(f"- **PCA Explained Variance (PC1-3)**: {pca_str}")

        if result.top_indicators:
            lines.append(f"- **Top Indicators**: {', '.join(result.top_indicators)}")
        lines.append("")

        # Interpretation
        lines.append("#### Interpretation")
        lines.append("")
        lines.append(result.interpretation)
        lines.append("")

        # Failure modes
        failure_modes = group_info.get("failure_modes", [])
        if failure_modes:
            lines.append("#### Potential Failure Modes")
            lines.append("")
            for fm in failure_modes:
                lines.append(f"- {fm}")
            lines.append("")

        # Confidence
        lines.append(f"#### Confidence Evaluation: **{result.confidence.upper()}**")
        lines.append("")

        if not result.success:
            lines.append(f"> **Error**: {result.error_message}")
            lines.append("")

        # Plot references
        lines.append("#### Visualizations")
        lines.append("")
        lines.append(f"- `{name}_timeseries.png` - Time series overview")
        lines.append(f"- `{name}_pca_scatter.png` - PCA projection")
        lines.append(f"- `{name}_correlation.png` - Correlation matrix")
        lines.append(f"- `{name}_lens_agreement.png` - Lens ranking comparison")
        lines.append(f"- `{name}_mrf_timeline.png` - MRF timeline")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- **Analysis Window**: {ANALYSIS_CONFIG['window_days']} days")
    lines.append(f"- **Step Size**: {ANALYSIS_CONFIG['step_days']} days")
    lines.append(f"- **Lenses Used**: {', '.join(ANALYSIS_CONFIG['lenses'])}")
    lines.append("")
    lines.append("Each dataset is processed through the PRISM lens pipeline. Rankings are computed")
    lines.append("per lens, then aggregated to compute cross-lens agreement (Kendall's tau) and")
    lines.append("mean rank fluctuation (MRF) metrics.")
    lines.append("")

    # Conclusions
    lines.append("## Conclusions")
    lines.append("")

    n_detected = sum(1 for r in results.results.values() if r.structure_detected)
    n_high_conf = sum(1 for r in results.results.values() if r.confidence == "high")

    lines.append(f"The PRISM engine detected structure in **{n_detected}/{results.datasets_analyzed}** datasets.")
    lines.append(f"High-confidence detections: **{n_high_conf}/{results.datasets_analyzed}**")
    lines.append("")

    # Check specific expectations
    if "pure_noise" in results.results:
        noise_result = results.results["pure_noise"]
        if not noise_result.structure_detected:
            lines.append("- **pure_noise**: Correctly identified as unstructured (no false positives)")
        else:
            lines.append("- **pure_noise**: WARNING - Structure falsely detected in noise baseline")

    if "clear_leader" in results.results:
        leader_result = results.results["clear_leader"]
        if leader_result.structure_detected and leader_result.confidence in ["medium", "high"]:
            lines.append("- **clear_leader**: Successfully detected dominant indicator structure")
        else:
            lines.append("- **clear_leader**: Leader detection below expected confidence")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by PRISM Benchmark Suite v1.0*")

    return "\n".join(lines)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_benchmark_suite(verbose: bool = True) -> BenchmarkSuiteResults:
    """
    Run the full benchmark suite.

    Args:
        verbose: Print progress messages

    Returns:
        BenchmarkSuiteResults with all analysis results
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Create output directory
    output_base = PROJECT_ROOT / "runs" / "benchmarks" / timestamp
    output_base.mkdir(parents=True, exist_ok=True)

    results = BenchmarkSuiteResults(
        timestamp=timestamp,
        output_dir=str(output_base),
    )

    if verbose:
        print("=" * 70)
        print("   PRISM BENCHMARK SUITE")
        print("=" * 70)
        print(f"   Timestamp: {timestamp}")
        print(f"   Output: {output_base}")
        print("=" * 70)
        print()

    total_start = time.time()

    # Get available benchmark groups
    available_groups = get_available_benchmark_groups()

    if not available_groups:
        print("WARNING: No benchmark data found in database.")
        print("Run: python prism_run.py --load-benchmarks")
        return results

    if verbose:
        print(f"Found {len(available_groups)} benchmark datasets: {available_groups}")
        print()

    # Analyze each dataset
    all_plots = []

    for group_name in available_groups:
        if verbose:
            print(f"Analyzing: {group_name}...")

        # Run analysis
        analysis = analyze_dataset(group_name, ANALYSIS_CONFIG)
        results.results[group_name] = analysis
        results.datasets_analyzed += 1

        if verbose:
            status = "OK" if analysis.success else "FAILED"
            print(f"  Status: {status} ({analysis.runtime_seconds:.1f}s)")
            print(f"  Lens Agreement: {analysis.lens_agreement:.2f}")
            print(f"  Structure Detected: {analysis.structure_detected}")
            print()

        # Generate plots
        if analysis.success:
            plots = generate_dataset_plots(group_name, analysis, output_base)
            all_plots.extend(plots)
            if verbose:
                print(f"  Generated {len(plots)} plots")
                print()

    results.total_runtime_seconds = time.time() - total_start

    # Generate summary metrics
    if results.results:
        results.summary_metrics = {
            "mean_lens_agreement": np.mean([r.lens_agreement for r in results.results.values()]),
            "mean_mrf": np.mean([r.mrf_score for r in results.results.values()]),
            "structure_detection_rate": sum(1 for r in results.results.values() if r.structure_detected) / len(results.results),
            "high_confidence_rate": sum(1 for r in results.results.values() if r.confidence == "high") / len(results.results),
        }

    # Generate report
    if verbose:
        print("Generating Markdown report...")

    report_content = generate_markdown_report(results)
    report_path = output_base / "report.md"
    report_path.write_text(report_content, encoding="utf-8")
    results.report_path = str(report_path)

    # Save summary JSON
    summary_path = output_base / "summary.json"
    summary_data = {
        "timestamp": results.timestamp,
        "datasets_analyzed": results.datasets_analyzed,
        "total_runtime_seconds": results.total_runtime_seconds,
        "summary_metrics": results.summary_metrics,
        "results": {name: asdict(r) for name, r in results.results.items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Create ZIP archive
    if verbose:
        print("Creating ZIP archive...")

    zip_path = output_base.parent / f"benchmarks_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in output_base.iterdir():
            zf.write(file, file.name)
    results.zip_path = str(zip_path)

    if verbose:
        print()
        print("=" * 70)
        print("   BENCHMARK SUITE COMPLETE")
        print("=" * 70)
        print(f"   Datasets: {results.datasets_analyzed}")
        print(f"   Runtime: {results.total_runtime_seconds:.1f}s")
        print(f"   Report: {results.report_path}")
        print(f"   ZIP: {results.zip_path}")
        print("=" * 70)

    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_benchmark_suite(verbose=True)

    if results.datasets_analyzed == 0:
        print("\nNo datasets analyzed. Load benchmarks first:")
        print("  python prism_run.py --load-benchmarks")
        sys.exit(1)

    sys.exit(0)
