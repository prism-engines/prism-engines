"""
PRISM Runner Package
=====================

Unified runner system for executing PRISM analyses.

Modes:
- CLI: Interactive menu-based interface
- Direct: Command-line arguments for scripting
- Web: Browser-based interface (Flask)
- HTML: Legacy HTML runner

Usage:
    # Interactive CLI
    python prism_run.py

    # Direct mode
    python prism_run.py --panel market --workflow regime_comparison

    # Web server
    python prism_run.py --web

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

# Import report generator (optional - requires jinja2)
try:
    from .report_generator import ReportGenerator, generate_report
    _REPORTS_AVAILABLE = True
except ImportError:
    _REPORTS_AVAILABLE = False
    ReportGenerator = None
    generate_report = None

# Import web server (optional - requires flask)
try:
    from .web_server import create_app, run_server
    _WEB_AVAILABLE = True
except ImportError:
    _WEB_AVAILABLE = False
    create_app = None
    run_server = None

__all__ = [
    "CLIRunner",
    "run_cli",
    "WorkflowExecutor",
    "OutputManager",
    "Dispatcher",
    "DispatchError",
]

if _REPORTS_AVAILABLE:
    __all__.extend([
        "ReportGenerator",
        "generate_report",
    ])

if _WEB_AVAILABLE:
    __all__.extend([
        "create_app",
        "run_server",
    ])
