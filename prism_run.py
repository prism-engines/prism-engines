#!/usr/bin/env python3
"""
PRISM Runner - Unified Entry Point
===================================

Launches the PRISM analysis interface.

Usage:
    python prism_run.py              # Opens HTML interface in browser
    python prism_run.py --cli        # Command-line interface
    python prism_run.py --port 8080  # Custom port
    python prism_run.py --no-browser # Don't auto-open browser

    python prism_run.py --workflow regime_comparison --panel market  # Direct run
    python prism_run.py --list-workflows   # List available workflows
    python prism_run.py --list-engines     # List available engines

The HTML runner provides:
- Panel selection (Market, Economy, Climate, Custom)
- Workflow selection (from registry)
- Timeframe configuration
- Engine weighting options
- One-click execution
- Auto-generated HTML reports
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Main entry point - delegates to runner module."""
    from runner.prism_run import main as runner_main
    runner_main()

if __name__ == "__main__":
    main()
