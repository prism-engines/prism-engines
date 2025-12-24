"""
PRISM Engines - Geometric analysis for time series data.

Quick Start:
    >>> import prism_engines as prism
    >>> results = prism.run("data.csv")
    >>> results.print_report()
    >>> results.plot()

Or step by step:
    >>> from prism_engines import load_csv, run_engines
    >>> df = load_csv("data.csv")
    >>> results = run_engines(df, engines=["pca", "correlation"])
    >>> print(results["pca"].metrics)
"""

__version__ = "0.1.0"

from .data import load_csv, validate_data
from .runner import run_engines, Results
from .engines import (
    BaseEngine,
    EngineResult,
    CorrelationEngine,
    PCAEngine,
    HurstEngine,
    list_engines,
    get_engine,
)


def run(data, engines=None):
    """
    Run PRISM analysis on data.

    This is the main entry point for quick analysis.

    Args:
        data: Path to CSV file or pandas DataFrame
        engines: List of engines to run (default: all)

    Returns:
        Results object with .report(), .plot(), .save() methods

    Example:
        >>> import prism_engines as prism
        >>> results = prism.run("my_data.csv")
        >>> results.print_report()
    """
    return run_engines(data, engines)


__all__ = [
    # Main API
    "run",
    "run_engines",
    "Results",
    # Data
    "load_csv",
    "validate_data",
    # Engines
    "BaseEngine",
    "EngineResult",
    "CorrelationEngine",
    "PCAEngine",
    "HurstEngine",
    "list_engines",
    "get_engine",
    # Version
    "__version__",
]
