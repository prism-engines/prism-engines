"""
PRISM Runner Package
=====================

Unified runner system for executing PRISM analyses.

Modes:
- CLI: Interactive menu-based interface
- Direct: Command-line arguments for scripting
- HTML: Web-based interface

Usage:
    # Interactive mode
    python prism_run.py

    # Direct mode
    python prism_run.py --panel market --workflow regime_comparison

    # List available options
    python prism_run.py --list-panels
    python prism_run.py --list-workflows

    # From code
    from runner import CLIRunner, WorkflowExecutor, OutputManager

    runner = CLIRunner()
    runner.run_interactive()

    # Or use executor directly
    executor = WorkflowExecutor()
    results = executor.execute("market", "regime_comparison")
"""

from .cli_runner import CLIRunner, run_cli
from .executor import WorkflowExecutor
from .output_manager import OutputManager
from .dispatcher import Dispatcher, DispatchError

__all__ = [
    "CLIRunner",
    "run_cli",
    "WorkflowExecutor",
    "OutputManager",
    "Dispatcher",
    "DispatchError",
]
