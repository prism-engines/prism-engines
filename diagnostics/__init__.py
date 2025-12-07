"""
PRISM Diagnostics Module

A unified framework for running system diagnostics, health checks,
and performance validation across the PRISM engine.

Usage:
    from diagnostics import DiagnosticRunner, run_all_diagnostics

    # Run all diagnostics
    results = run_all_diagnostics()

    # Run specific category
    results = run_all_diagnostics(categories=['health', 'performance'])

    # Use the runner directly
    runner = DiagnosticRunner()
    runner.register_diagnostic(MyCustomDiagnostic())
    results = runner.run_all()
"""

from diagnostics.core.base import (
    BaseDiagnostic,
    DiagnosticStatus,
    DiagnosticResult,
    DiagnosticCategory,
)
from diagnostics.core.runner import DiagnosticRunner
from diagnostics.core.registry import DiagnosticRegistry, get_registry
from diagnostics.core.report import DiagnosticReporter

# Convenience function
def run_all_diagnostics(
    categories: list = None,
    quick_mode: bool = False,
    verbose: bool = True
) -> dict:
    """
    Run all registered diagnostics.

    Args:
        categories: List of categories to run (None = all)
        quick_mode: If True, skip slow diagnostics
        verbose: If True, print progress

    Returns:
        Dictionary with diagnostic results and summary
    """
    runner = DiagnosticRunner(verbose=verbose)
    return runner.run_all(categories=categories, quick_mode=quick_mode)


__all__ = [
    'BaseDiagnostic',
    'DiagnosticStatus',
    'DiagnosticResult',
    'DiagnosticCategory',
    'DiagnosticRunner',
    'DiagnosticRegistry',
    'DiagnosticReporter',
    'get_registry',
    'run_all_diagnostics',
]

__version__ = '1.0.0'
