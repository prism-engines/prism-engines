#!/usr/bin/env python3
"""
PRISM Engine - Unified Entry Point
====================================

Usage:
    python prism_run.py                    # Interactive CLI mode
    python prism_run.py --web              # Web server mode
    python prism_run.py --panel market     # Direct mode
    python prism_run.py --html             # Legacy HTML runner
    python prism_run.py --benchmarks       # Run benchmark test suite
    python prism_run.py --help             # Show help

Web Server Options:
    python prism_run.py --web              # Start on default port 5000
    python prism_run.py --web --port 8080  # Custom port
    python prism_run.py --web --debug      # Debug mode with auto-reload

CLI Options:
    python prism_run.py --list-panels      # List available panels
    python prism_run.py --list-workflows   # List available workflows
    python prism_run.py --list-engines     # List available engines
    python prism_run.py --load-benchmarks  # Load benchmark datasets into DB

Benchmark Suite:
    python prism_run.py --benchmarks       # Run full benchmark test suite
                                           # (3-minute hybrid analysis)

Direct Mode:
    python prism_run.py --panel market --workflow regime_comparison
    python prism_run.py -p market -w daily_update --verbose

See runner/ for implementation details.
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Main entry point with mode routing."""

    # Check for --benchmarks flag (benchmark test suite)
    if "--benchmarks" in sys.argv:
        try:
            from workflows.benchmark_suite import run_benchmark_suite
            print("Starting PRISM Benchmark Suite...")
            print()
            results = run_benchmark_suite(verbose=True)
            if results.datasets_analyzed == 0:
                print("\nNo datasets analyzed. Load benchmarks first:")
                print("  python prism_run.py --load-benchmarks")
                return 1
            return 0
        except Exception as e:
            print(f"Error running benchmark suite: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Check for --load-benchmarks flag (benchmark loader)
    if "--load-benchmarks" in sys.argv:
        try:
            from data.sql.load_benchmarks import load_all_benchmarks
            load_all_benchmarks()
            return 0
        except Exception as e:
            print(f"Error loading benchmarks: {e}")
            return 1

    # Check for --web flag (web server mode)
    if "--web" in sys.argv:
        try:
            from runner.web_server import run_server

            # Parse optional port
            port = 5000
            if "--port" in sys.argv:
                idx = sys.argv.index("--port")
                if idx + 1 < len(sys.argv):
                    try:
                        port = int(sys.argv[idx + 1])
                    except ValueError:
                        print(f"Invalid port: {sys.argv[idx + 1]}")
                        return 1

            debug = "--debug" in sys.argv
            return run_server(port=port, debug=debug)

        except ImportError as e:
            print(f"Web server not available: {e}")
            print("Install Flask: pip install flask")
            return 1

    # Check for --html flag (legacy HTML runner)
    if "--html" in sys.argv:
        try:
            from runner.prism_run import main as html_main
            return html_main()
        except ImportError:
            print("HTML runner not available. Use CLI mode instead.")
            print("Run: python prism_run.py --help")
            return 1

    # Default: CLI runner
    try:
        from runner.cli_runner import run_cli
        return run_cli()
    except ImportError as e:
        print(f"CLI runner not available: {e}")
        # Fallback to HTML runner
        try:
            from runner.prism_run import main as html_main
            return html_main()
        except ImportError:
            print("No runners available")
            return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
