"""
Report Builder for PRISM Engine Benchmarks

Generates clean, academic-style Markdown reports from benchmark results.

Reports generated:
- Benchmark_Results.md: Master summary with links to detailed reports
- Engine_Correctness.md: Synthetic scenario correctness tests
- Engine_Performance.md: Performance benchmarks with timings
- Overnight_Analysis_Benchmark.md: Overnight script benchmarks
- ML_Seismometer_Benchmark.md: ML component benchmarks
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import subprocess

from .engine_correctness import CorrectnessResult
from .engine_performance import PerformanceResult
from .overnight_benchmarks import OvernightResult
from .ml_seismometer_benchmarks import MLBenchmarkResult


# Default output directory
DEFAULT_REPORT_DIR = Path(__file__).parent.parent / "reports" / "benchmarks"


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_date() -> str:
    """Get current date."""
    return datetime.now().strftime("%Y-%m-%d")


def _write_markdown(path: Path, content: str) -> None:
    """Write content to a markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _format_seconds(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def _format_mb(mb: float) -> str:
    """Format megabytes with appropriate precision."""
    if mb < 1:
        return f"{mb:.3f} MB"
    elif mb < 100:
        return f"{mb:.1f} MB"
    else:
        return f"{mb:.0f} MB"


def _build_header(title: str, db_path: str = "") -> str:
    """Build a standard report header."""
    lines = [
        f"# {title}",
        "",
        f"- **Date**: {_get_date()}",
        f"- **Commit**: `{_get_git_commit()}`",
    ]
    if db_path:
        lines.append(f"- **Database**: `{db_path}`")
    lines.extend([
        f"- **Generated**: {_get_timestamp()}",
        "",
    ])
    return "\n".join(lines)


def build_master_report(
    correctness: list[CorrectnessResult] | None = None,
    performance: list[PerformanceResult] | None = None,
    overnight: list[OvernightResult] | None = None,
    ml: list[MLBenchmarkResult] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Build the master benchmark report.

    This is the top-level summary with links to detailed reports.
    """
    output_dir = output_dir or DEFAULT_REPORT_DIR
    output_path = output_dir / "Benchmark_Results.md"

    sections = []

    # Header
    sections.append(_build_header("PRISM Engine Benchmark Results"))

    # Overview
    sections.append("## Overview\n")
    overview_lines = [
        "This document summarizes the benchmark results for the PRISM Engine.",
        "The benchmarks cover four main areas:",
        "",
        "1. **Engine Correctness**: Validates lens behavior on synthetic signals",
        "2. **Engine Performance**: Measures temporal analysis throughput",
        "3. **Overnight Analysis**: Tests overnight script components",
        "4. **ML/Seismometer**: Validates ML pipeline components",
        "",
    ]
    sections.append("\n".join(overview_lines))

    # Quick Summary Table
    sections.append("## Quick Summary\n")

    table_lines = [
        "| Suite | Status | Tests/Benchmarks | Runtime | Notes |",
        "|-------|--------|------------------|---------|-------|",
    ]

    # Correctness summary
    if correctness:
        n_passed = sum(1 for r in correctness if r.success)
        total_checks = sum(len(r.checks) for r in correctness)
        passed_checks = sum(sum(r.checks.values()) for r in correctness)
        total_runtime = sum(r.runtime_seconds for r in correctness)
        status = "PASS" if n_passed == len(correctness) else "PARTIAL"
        table_lines.append(
            f"| Correctness | {status} | {len(correctness)} scenarios, "
            f"{passed_checks}/{total_checks} checks | {_format_seconds(total_runtime)} | "
            f"Synthetic signal validation |"
        )
    else:
        table_lines.append("| Correctness | - | Not run | - | - |")

    # Performance summary
    if performance:
        n_passed = sum(1 for r in performance if r.success)
        total_runtime = sum(r.runtime_seconds for r in performance)
        status = "PASS" if n_passed == len(performance) else "PARTIAL"
        table_lines.append(
            f"| Performance | {status} | {len(performance)} profiles | "
            f"{_format_seconds(total_runtime)} | Temporal analysis benchmarks |"
        )
    else:
        table_lines.append("| Performance | - | Not run | - | - |")

    # Overnight summary
    if overnight:
        n_passed = sum(1 for r in overnight if r.success)
        total_runtime = sum(r.runtime_seconds for r in overnight)
        status = "PASS" if n_passed == len(overnight) else "PARTIAL"
        scripts = ", ".join(r.mode for r in overnight)
        table_lines.append(
            f"| Overnight | {status} | {len(overnight)} modes ({scripts}) | "
            f"{_format_seconds(total_runtime)} | Script component tests |"
        )
    else:
        table_lines.append("| Overnight | - | Not run | - | - |")

    # ML summary
    if ml:
        n_passed = sum(1 for r in ml if r.success)
        total_runtime = sum(r.runtime_seconds for r in ml)
        status = "PASS" if n_passed == len(ml) else "PARTIAL"
        components = ", ".join(r.component for r in ml)
        table_lines.append(
            f"| ML/Seismometer | {status} | {len(ml)} components | "
            f"{_format_seconds(total_runtime)} | {components} |"
        )
    else:
        table_lines.append("| ML/Seismometer | - | Not run | - | - |")

    sections.append("\n".join(table_lines))
    sections.append("")

    # Detailed Report Links
    sections.append("## Detailed Reports\n")
    sections.append("- [Engine Correctness](Engine_Correctness.md)")
    sections.append("- [Engine Performance](Engine_Performance.md)")
    sections.append("- [Overnight Analysis](Overnight_Analysis_Benchmark.md)")
    sections.append("- [ML/Seismometer](ML_Seismometer_Benchmark.md)")
    sections.append("")

    # Interpretation
    sections.append("## Interpretation\n")
    interpretation_lines = []

    if correctness:
        n_passed = sum(1 for r in correctness if r.success)
        if n_passed == len(correctness):
            interpretation_lines.append(
                "- **Correctness**: All scenarios passed. Lenses correctly detect "
                "expected signal properties in synthetic data."
            )
        else:
            n_failed = len(correctness) - n_passed
            interpretation_lines.append(
                f"- **Correctness**: {n_failed} scenario(s) had issues. "
                "See detailed report for specifics."
            )

    if performance:
        fastest = min(performance, key=lambda r: r.runtime_seconds)
        slowest = max(performance, key=lambda r: r.runtime_seconds)
        interpretation_lines.append(
            f"- **Performance**: Fastest profile: {fastest.name} ({_format_seconds(fastest.runtime_seconds)}), "
            f"Slowest: {slowest.name} ({_format_seconds(slowest.runtime_seconds)})"
        )

    if ml:
        n_passed = sum(1 for r in ml if r.success)
        if n_passed == len(ml):
            interpretation_lines.append(
                "- **ML/Seismometer**: All components operational. "
                "Pipeline runs end-to-end without errors."
            )
        else:
            interpretation_lines.append(
                "- **ML/Seismometer**: Some components had issues. "
                "See detailed report for specifics."
            )

    if interpretation_lines:
        sections.append("\n".join(interpretation_lines))
    else:
        sections.append("*No benchmarks were run.*")

    sections.append("")

    # Footer
    sections.append("---")
    sections.append(f"*Report generated by PRISM Engine Benchmark Harness v1.0*")

    content = "\n".join(sections)
    _write_markdown(output_path, content)

    return output_path


def build_correctness_report(
    results: list[CorrectnessResult],
    output_dir: Path | None = None,
) -> Path:
    """Build the engine correctness benchmark report."""
    output_dir = output_dir or DEFAULT_REPORT_DIR
    output_path = output_dir / "Engine_Correctness.md"

    sections = []

    # Header
    sections.append(_build_header("Engine Correctness Benchmarks"))

    # Configuration
    sections.append("## Configuration\n")
    if results:
        example = results[0]
        config_lines = [
            f"- **Lenses tested**: {', '.join(example.lenses_run)}",
            f"- **Scenarios**: {len(results)}",
            f"- **Data type**: Synthetic signals",
        ]
        sections.append("\n".join(config_lines))
    sections.append("")

    # Methods
    sections.append("## Methods\n")
    sections.append(
        "Each synthetic scenario generates data with known statistical properties. "
        "The temporal engine and analytical lenses process the data, then validation "
        "checks confirm expected behaviors are detected.\n"
    )
    sections.append(
        "Scenarios tested:\n"
        "- **Sine wave**: Periodic signal detection\n"
        "- **Trend + noise**: Trend and decomposition detection\n"
        "- **Step change**: Regime shift detection\n"
        "- **White noise**: Null baseline (no structure)\n"
        "- **Correlated pairs**: Correlation and influence detection\n"
        "- **Mean reverting**: Stationarity detection\n"
        "- **Random walk**: Non-stationary behavior\n"
    )

    # Results Table
    sections.append("## Results\n")

    table_lines = [
        "| Scenario | Rows | Cols | Lenses | Checks Passed | Runtime | Status |",
        "|----------|------|------|--------|---------------|---------|--------|",
    ]

    for r in results:
        checks_passed = sum(r.checks.values())
        checks_total = len(r.checks)
        status = "PASS" if r.success and checks_passed == checks_total else "PARTIAL" if r.success else "FAIL"
        table_lines.append(
            f"| {r.scenario_name} | {r.n_rows} | {r.n_columns} | "
            f"{len(r.lenses_run)} | {checks_passed}/{checks_total} | "
            f"{_format_seconds(r.runtime_seconds)} | {status} |"
        )

    sections.append("\n".join(table_lines))
    sections.append("")

    # Detailed Results
    sections.append("## Detailed Results\n")

    for r in results:
        sections.append(f"### {r.scenario_name.replace('_', ' ').title()}\n")

        # Metrics
        if r.metrics:
            sections.append("**Metrics:**\n")
            for key, value in r.metrics.items():
                if isinstance(value, float):
                    sections.append(f"- {key}: {value:.4f}")
                else:
                    sections.append(f"- {key}: {value}")
            sections.append("")

        # Checks
        if r.checks:
            sections.append("**Validation Checks:**\n")
            for check, passed in r.checks.items():
                status_emoji = "PASS" if passed else "FAIL"
                sections.append(f"- {check}: {status_emoji}")
            sections.append("")

        # Notes
        if r.notes:
            sections.append("**Notes:**\n")
            for note in r.notes:
                sections.append(f"- {note}")
            sections.append("")

    # Summary Statistics
    sections.append("## Summary Statistics\n")

    total_checks = sum(len(r.checks) for r in results)
    passed_checks = sum(sum(r.checks.values()) for r in results)
    total_runtime = sum(r.runtime_seconds for r in results)

    sections.append(f"- **Total scenarios**: {len(results)}")
    sections.append(f"- **Total checks**: {passed_checks}/{total_checks} passed")
    sections.append(f"- **Total runtime**: {_format_seconds(total_runtime)}")
    sections.append(f"- **Pass rate**: {100 * passed_checks / total_checks:.1f}%")
    sections.append("")

    content = "\n".join(sections)
    _write_markdown(output_path, content)

    return output_path


def build_performance_report(
    results: list[PerformanceResult],
    output_dir: Path | None = None,
) -> Path:
    """Build the engine performance benchmark report."""
    output_dir = output_dir or DEFAULT_REPORT_DIR
    output_path = output_dir / "Engine_Performance.md"

    sections = []

    # Get DB path from first result
    db_path = ""
    for r in results:
        for note in r.notes:
            if note.startswith("DB:"):
                db_path = note[4:].strip()
                break
        if db_path:
            break

    # Header
    sections.append(_build_header("Engine Performance Benchmarks", db_path=db_path))

    # Configuration
    sections.append("## Configuration\n")

    from .engine_performance import BENCHMARK_PROFILES

    sections.append("**Benchmark Profiles:**\n")
    for name, config in BENCHMARK_PROFILES.items():
        sections.append(f"- **{name}**: {config['description']}")
        sections.append(f"  - Lenses: {', '.join(config['lenses'])}")
        sections.append(f"  - Window: {config['window_days']} days, Step: {config['step_days']} days")
    sections.append("")

    # Methods
    sections.append("## Methods\n")
    sections.append(
        "Performance benchmarks load real data from the database and run "
        "temporal analysis with different configurations. Each profile tests "
        "a different combination of lenses, window sizes, and step intervals.\n"
    )
    sections.append(
        "Metrics measured:\n"
        "- **Runtime**: Wall-clock time for full analysis\n"
        "- **Panel size**: Memory footprint of input data\n"
        "- **Result size**: Memory footprint of output\n"
        "- **Windows**: Number of rolling windows analyzed\n"
    )

    # Results Table
    sections.append("## Results\n")

    table_lines = [
        "| Profile | Indicators | Lenses | Windows | Runtime | Panel | Result | Status |",
        "|---------|------------|--------|---------|---------|-------|--------|--------|",
    ]

    for r in results:
        status = "PASS" if r.success else "FAIL"
        table_lines.append(
            f"| {r.name} | {r.n_indicators} | {r.n_lenses} | {r.n_windows} | "
            f"{_format_seconds(r.runtime_seconds)} | {_format_mb(r.panel_size_mb)} | "
            f"{_format_mb(r.result_size_mb)} | {status} |"
        )

    sections.append("\n".join(table_lines))
    sections.append("")

    # Detailed Results
    sections.append("## Detailed Results\n")

    for r in results:
        sections.append(f"### {r.name}\n")

        sections.append(f"- **Profile**: {r.profile}")
        sections.append(f"- **Indicators**: {r.n_indicators}")
        sections.append(f"- **Lenses**: {', '.join(r.lenses_used)}")
        sections.append(f"- **Windows analyzed**: {r.n_windows}")
        sections.append(f"- **Runtime**: {_format_seconds(r.runtime_seconds)}")
        sections.append(f"- **Panel size**: {_format_mb(r.panel_size_mb)}")
        sections.append(f"- **Result size**: {_format_mb(r.result_size_mb)}")

        if r.date_range:
            sections.append(f"- **Date range**: {r.date_range[0]} to {r.date_range[1]}")

        if r.notes:
            sections.append("\n**Notes:**")
            for note in r.notes:
                sections.append(f"- {note}")

        if r.error_message:
            sections.append(f"\n**Error**: {r.error_message}")

        sections.append("")

    # Interpretation
    sections.append("## Interpretation\n")

    if results:
        fastest = min(results, key=lambda r: r.runtime_seconds)
        slowest = max(results, key=lambda r: r.runtime_seconds)

        sections.append(f"- **Fastest**: {fastest.name} ({_format_seconds(fastest.runtime_seconds)})")
        sections.append(f"- **Slowest**: {slowest.name} ({_format_seconds(slowest.runtime_seconds)})")

        if fastest.n_windows > 0 and slowest.n_windows > 0:
            fastest_rate = fastest.n_windows / fastest.runtime_seconds
            slowest_rate = slowest.n_windows / slowest.runtime_seconds
            sections.append(f"- **Throughput range**: {fastest_rate:.1f} - {slowest_rate:.1f} windows/sec")

        # Memory efficiency
        total_panel_mb = sum(r.panel_size_mb for r in results)
        total_result_mb = sum(r.result_size_mb for r in results)
        if total_panel_mb > 0:
            expansion_ratio = total_result_mb / total_panel_mb
            sections.append(f"- **Average result/panel ratio**: {expansion_ratio:.2f}x")

    sections.append("")

    content = "\n".join(sections)
    _write_markdown(output_path, content)

    return output_path


def build_overnight_report(
    results: list[OvernightResult],
    output_dir: Path | None = None,
) -> Path:
    """Build the overnight analysis benchmark report."""
    output_dir = output_dir or DEFAULT_REPORT_DIR
    output_path = output_dir / "Overnight_Analysis_Benchmark.md"

    sections = []

    # Header
    sections.append(_build_header("Overnight Analysis Benchmarks"))

    # Configuration
    sections.append("## Configuration\n")
    sections.append(
        "These benchmarks test the overnight analysis scripts by running "
        "their core components with reduced parameters for faster testing.\n"
    )
    sections.append(
        "**Scripts tested:**\n"
        "- `overnight_lite.py`: Fast version (30-60 min in production)\n"
        "- `overnight_analysis.py`: Full version (6-12 hours in production)\n"
    )

    # Methods
    sections.append("## Methods\n")
    sections.append(
        "The benchmarks exercise each major component of the overnight scripts:\n"
        "- Multi-resolution analysis (rolling windows at multiple time scales)\n"
        "- Bootstrap confidence intervals\n"
        "- Monte Carlo null distribution generation\n"
        "- HMM regime analysis (if hmmlearn available)\n"
        "- GMM regime detection\n"
    )

    # Results Table
    sections.append("## Results\n")

    table_lines = [
        "| Script | Mode | Indicators | Windows | Components | Runtime | Status |",
        "|--------|------|------------|---------|------------|---------|--------|",
    ]

    for r in results:
        status = "PASS" if r.success else "FAIL"
        n_components = len(r.components_run)
        table_lines.append(
            f"| {r.script_name} | {r.mode} | {r.n_indicators or '-'} | "
            f"{r.n_windows or '-'} | {n_components} | "
            f"{_format_seconds(r.runtime_seconds)} | {status} |"
        )

    sections.append("\n".join(table_lines))
    sections.append("")

    # Detailed Results
    sections.append("## Detailed Results\n")

    for r in results:
        sections.append(f"### {r.script_name} ({r.mode})\n")

        sections.append(f"- **Runtime**: {_format_seconds(r.runtime_seconds)}")
        sections.append(f"- **Indicators**: {r.n_indicators or 'N/A'}")
        sections.append(f"- **Windows**: {r.n_windows or 'N/A'}")
        sections.append(f"- **Parallel**: {'Yes' if r.parallel_enabled else 'No'}")
        if r.n_workers:
            sections.append(f"- **Workers**: {r.n_workers}")

        if r.components_run:
            sections.append(f"\n**Components tested:**")
            for comp in r.components_run:
                sections.append(f"- {comp}")

        if r.notes:
            sections.append(f"\n**Notes:**")
            for note in r.notes:
                sections.append(f"- {note}")

        if r.error_message:
            sections.append(f"\n**Error**: {r.error_message}")

        sections.append("")

    # Interpretation
    sections.append("## Interpretation\n")

    if results:
        all_passed = all(r.success for r in results)
        if all_passed:
            sections.append(
                "All overnight script components executed successfully. "
                "The benchmark parameters were reduced for testing speed, "
                "but the same code paths used in production were exercised."
            )
        else:
            failed = [r for r in results if not r.success]
            sections.append(
                f"{len(failed)} script(s) had issues. "
                "See error messages above for details."
            )

        # Component coverage
        all_components = set()
        for r in results:
            all_components.update(r.components_run)

        sections.append(f"\n**Component coverage**: {', '.join(sorted(all_components))}")

    sections.append("")

    content = "\n".join(sections)
    _write_markdown(output_path, content)

    return output_path


def build_ml_seismometer_report(
    results: list[MLBenchmarkResult],
    output_dir: Path | None = None,
) -> Path:
    """Build the ML/Seismometer benchmark report."""
    output_dir = output_dir or DEFAULT_REPORT_DIR
    output_path = output_dir / "ML_Seismometer_Benchmark.md"

    sections = []

    # Header
    sections.append(_build_header("ML / Seismometer Benchmarks"))

    # Configuration
    sections.append("## Configuration\n")
    sections.append(
        "These benchmarks test the ML components of the PRISM Engine:\n"
        "- **Seismometer**: Ensemble instability detection\n"
        "- **Regime Classification**: GMM, K-means, and HMM-based regime detection\n"
        "- **Detector Components**: Individual seismometer detectors\n"
    )

    # Methods
    sections.append("## Methods\n")
    sections.append(
        "Each ML component is tested end-to-end:\n"
        "1. Load feature data from the database\n"
        "2. Split into baseline (training) and evaluation periods\n"
        "3. Fit models on baseline data\n"
        "4. Score/predict on evaluation data\n"
        "5. Verify output structure and ranges\n"
    )

    # Results Table
    sections.append("## Results\n")

    table_lines = [
        "| Component | Samples | Features | Subcomponents | Runtime | Status |",
        "|-----------|---------|----------|---------------|---------|--------|",
    ]

    for r in results:
        status = "PASS" if r.success else "FAIL"
        n_subcomponents = len(r.subcomponents_tested)
        table_lines.append(
            f"| {r.component} | {r.n_samples or '-'} | {r.n_features or '-'} | "
            f"{n_subcomponents} | {_format_seconds(r.runtime_seconds)} | {status} |"
        )

    sections.append("\n".join(table_lines))
    sections.append("")

    # Detailed Results
    sections.append("## Detailed Results\n")

    for r in results:
        sections.append(f"### {r.component.replace('_', ' ').title()}\n")

        sections.append(f"- **Runtime**: {_format_seconds(r.runtime_seconds)}")
        sections.append(f"- **Samples**: {r.n_samples or 'N/A'}")
        sections.append(f"- **Features**: {r.n_features or 'N/A'}")

        if r.subcomponents_tested:
            sections.append(f"\n**Subcomponents tested:**")
            for sub in r.subcomponents_tested:
                sections.append(f"- {sub}")

        if r.output_snapshot:
            sections.append(f"\n**Output snapshot:**")
            for key, value in r.output_snapshot.items():
                if isinstance(value, float):
                    sections.append(f"- {key}: {value:.4f}")
                elif isinstance(value, dict) and len(str(value)) < 100:
                    sections.append(f"- {key}: {value}")
                elif isinstance(value, list) and len(value) <= 5:
                    sections.append(f"- {key}: {value}")
                else:
                    sections.append(f"- {key}: (complex value)")

        if r.notes:
            sections.append(f"\n**Notes:**")
            for note in r.notes:
                sections.append(f"- {note}")

        if r.error_message:
            sections.append(f"\n**Error**: {r.error_message}")

        sections.append("")

    # Seismometer Stability Index Guide
    sections.append("## Seismometer Stability Index Reference\n")
    sections.append(
        "The Seismometer produces a stability index from 0 to 1:\n\n"
        "| Range | Level | Interpretation |\n"
        "|-------|-------|----------------|\n"
        "| 0.90-1.00 | Stable | Normal market behavior |\n"
        "| 0.70-0.90 | Elevated | Increased noise, heightened attention |\n"
        "| 0.50-0.70 | Pre-instability | Structural stress emerging |\n"
        "| 0.30-0.50 | Divergence | Active structural breakdown |\n"
        "| 0.00-0.30 | High-risk | Imminent or ongoing instability |\n"
    )

    # Interpretation
    sections.append("## Interpretation\n")

    if results:
        all_passed = all(r.success for r in results)
        if all_passed:
            sections.append(
                "All ML components executed successfully. The pipeline is "
                "operational and producing output in expected ranges."
            )
        else:
            failed = [r for r in results if not r.success]
            sections.append(
                f"{len(failed)} component(s) had issues. "
                "See error messages above for details."
            )

        # Check for seismometer stability output
        for r in results:
            if r.component == "seismometer" and r.success:
                stability = r.output_snapshot.get("stability_index")
                alert = r.output_snapshot.get("alert_level")
                if stability is not None:
                    sections.append(
                        f"\n**Current seismometer status**: {alert} "
                        f"(stability index: {stability:.3f})"
                    )

    sections.append("")

    content = "\n".join(sections)
    _write_markdown(output_path, content)

    return output_path


def build_all_reports(
    correctness: list[CorrectnessResult] | None = None,
    performance: list[PerformanceResult] | None = None,
    overnight: list[OvernightResult] | None = None,
    ml: list[MLBenchmarkResult] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Build all benchmark reports.

    Parameters
    ----------
    correctness : list[CorrectnessResult], optional
        Correctness benchmark results.
    performance : list[PerformanceResult], optional
        Performance benchmark results.
    overnight : list[OvernightResult], optional
        Overnight benchmark results.
    ml : list[MLBenchmarkResult], optional
        ML/Seismometer benchmark results.
    output_dir : Path, optional
        Output directory for reports.

    Returns
    -------
    list[Path]
        Paths to all generated report files.
    """
    output_dir = output_dir or DEFAULT_REPORT_DIR
    generated_reports = []

    # Build individual reports
    if correctness:
        path = build_correctness_report(correctness, output_dir)
        generated_reports.append(path)

    if performance:
        path = build_performance_report(performance, output_dir)
        generated_reports.append(path)

    if overnight:
        path = build_overnight_report(overnight, output_dir)
        generated_reports.append(path)

    if ml:
        path = build_ml_seismometer_report(ml, output_dir)
        generated_reports.append(path)

    # Build master report (always)
    master_path = build_master_report(
        correctness=correctness,
        performance=performance,
        overnight=overnight,
        ml=ml,
        output_dir=output_dir,
    )
    generated_reports.insert(0, master_path)

    return generated_reports
