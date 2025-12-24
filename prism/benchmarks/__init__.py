"""
PRISM Benchmarks Module

Compare PRISM against traditional quantitative methods.

Usage:
    from prism.benchmarks import run_full_benchmark, TraditionalAnalysis
    
    # Run traditional methods
    traditional = TraditionalAnalysis()
    trad_results = traditional.full_analysis(df, vix=vix_series)
    
    # Compare against PRISM
    results = run_full_benchmark(
        df=df,
        prism_engine_results=prism_results,
        vix=vix_series,
    )
    
    print(results["report"])
"""

from .traditional_methods import (
    TraditionalAnalysis,
    ThresholdRegimes,
    MarkovSwitchingBaseline,
    RollingCorrelation,
    MovingAverageTrend,
    SimpleCrossCorrelation,
)

from .compare_prism_traditional import (
    PRISMBenchmark,
    BenchmarkResult,
    KNOWN_EVENTS,
    run_full_benchmark,
)


__all__ = [
    # Traditional methods
    "TraditionalAnalysis",
    "ThresholdRegimes",
    "MarkovSwitchingBaseline",
    "RollingCorrelation",
    "MovingAverageTrend",
    "SimpleCrossCorrelation",
    
    # Comparison framework
    "PRISMBenchmark",
    "BenchmarkResult",
    "KNOWN_EVENTS",
    "run_full_benchmark",
]
