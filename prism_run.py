#!/usr/bin/env python3
"""
PRISM Engine - Main Entry Point
=================================

Unified runner for PRISM analysis system.

Usage:
    python prism_run.py                    # Interactive CLI mode
    python prism_run.py --panel market     # Direct mode
    python prism_run.py --html             # HTML mode (if available)
    python prism_run.py --help             # Show help

CLI Options:
    python prism_run.py --list-panels      # List available panels
    python prism_run.py --list-workflows   # List available workflows
    python prism_run.py --list-engines     # List available engines

Direct Mode:
    python prism_run.py --panel market --workflow regime_comparison
    python prism_run.py -p market -w daily_update --verbose

See runner/ for implementation details.
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Main entry point."""
    # Quick check for --html flag before full argument parsing
    if "--html" in sys.argv:
        try:
            from runner.prism_run import main as html_main
            return html_main()
        except ImportError:
            print("HTML runner not available. Use CLI mode instead.")
            print("Run: python prism_run.py --help")
            return 1

    # Default to CLI mode
    try:
        from runner.cli_runner import run_cli
        return run_cli()
    except ImportError as e:
        print(f"Error importing CLI runner: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install pyyaml pandas")
        return 1


if __name__ == "__main__":
    sys.exit(main())
