"""
PRISM Engine Benchmark Harness

A comprehensive benchmarking suite for the PRISM Engine that exercises:
- Temporal engine and rolling analysis
- Analytical lenses (correctness and performance)
- Overnight analysis scripts
- ML/Seismometer components

All benchmarks produce structured result objects that feed into
automated Markdown report generation.
"""

from .synthetic_scenarios import (
    generate_sine_wave,
    generate_trend_noise,
    generate_step_change,
    generate_white_noise,
    generate_correlated_pairs,
    SyntheticScenario,
)

from .engine_correctness import (
    CorrectnessResult,
    run_engine_correctness_benchmarks,
)

from .engine_performance import (
    PerformanceResult,
    run_engine_performance_benchmarks,
)

from .overnight_benchmarks import (
    OvernightResult,
    run_overnight_benchmarks,
)

from .ml_seismometer_benchmarks import (
    MLBenchmarkResult,
    run_ml_seismometer_benchmarks,
)

from .report_builder import (
    build_all_reports,
    build_master_report,
    build_correctness_report,
    build_performance_report,
    build_overnight_report,
    build_ml_seismometer_report,
)

__version__ = "1.0.0"
__all__ = [
    # Synthetic data
    "generate_sine_wave",
    "generate_trend_noise",
    "generate_step_change",
    "generate_white_noise",
    "generate_correlated_pairs",
    "SyntheticScenario",
    # Results
    "CorrectnessResult",
    "PerformanceResult",
    "OvernightResult",
    "MLBenchmarkResult",
    # Benchmark runners
    "run_engine_correctness_benchmarks",
    "run_engine_performance_benchmarks",
    "run_overnight_benchmarks",
    "run_ml_seismometer_benchmarks",
    # Report builders
    "build_all_reports",
    "build_master_report",
    "build_correctness_report",
    "build_performance_report",
    "build_overnight_report",
    "build_ml_seismometer_report",
]
